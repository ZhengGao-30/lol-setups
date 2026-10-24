"""TAGNet baseline (Zhang et al., IEEE ToG 2026) ported to the LoL V layer, plus its counting baseline.

    python -m lolsetup.baseline_tagnet fit --variant ctx|noctx --seed 3     # one (variant, seed) -> fits/baseline/
    python -m lolsetup.baseline_tagnet aggregate                           # -> results/baseline/

Port of the published design: each pre-spawn snapshot (s-90..s every 15 s, 7 snapshots in place of their 5) is a
fully connected graph on the 5 players (node = zone one-hot, alive flag, role one-hot, position; edge =
Euclidean distance); a 3-layer distance-biased GAT (dim 64, one head) is trained by self-supervised reconstruction
(zone occupancy 1.0, pit/base occupancy 0.5, the 10 pairwise distances 1.0; AdamW, <= 40 epochs, series-grouped
70/15/15 split); mean-pooled snapshot embeddings, concatenated with a random-projected context vector (gold
difference, side, time, objective type; zero in the 'noctx' variant), go through a fixed untrained GRU
(hidden 64); K-Means (20 inits) on the final state with K = 6, and K = 2..15 chosen by the mean dense rank of
silhouette, Davies-Bouldin and Calinski-Harabasz. Counting baseline: zone-occupancy counts per snapshot
(7 x 7) -> K-Means K = 6.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit

from . import io, vdata

FIT = io.ROOT / "fits" / "baseline"
OUT = io.RESULTS / "baseline"
OBJ = ["DRAGON", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR", "ELDER"]
K_TAG = 6
IU = np.triu_indices(5, 1)


class GATLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = torch.nn.Linear(d, d, bias=False)
        self.a_src, self.a_dst = torch.nn.Linear(d, 1, bias=False), torch.nn.Linear(d, 1, bias=False)
        self.bias = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))  # distance bias
        self.ln1, self.ln2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.ReLU(), torch.nn.Linear(2 * d, d))

    def forward(self, h, dist):                              # h: B x 5 x d, dist: B x 5 x 5
        z = self.W(h)
        e = torch.nn.functional.leaky_relu(self.a_src(z) + self.a_dst(z).transpose(1, 2), 0.2) + self.bias(dist[..., None]).squeeze(-1)
        h = self.ln1(h + torch.softmax(e, -1) @ z)
        return self.ln2(h + self.ffn(h))


class Encoder(torch.nn.Module):
    def __init__(self, f_in, d=64):
        super().__init__()
        self.inp = torch.nn.Linear(f_in, d)
        self.layers = torch.nn.ModuleList([GATLayer(d) for _ in range(3)])
        self.dec = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 7 + 2 + 10))

    def embed(self, x, pos):
        dist = torch.cdist(pos, pos)
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h, dist)
        return h.mean(1)

    def forward(self, x, pos):
        e = self.embed(x, pos)
        return e, self.dec(e)


def targets(X, P):
    """Reconstruction targets per snapshot: zone occupancy shares (7), pit and base shares (2), 10 pairwise distances."""
    alive = X[..., 7]
    zone = (X[..., :7] * alive[..., None]).sum(-2) / np.maximum(alive.sum(-1, keepdims=True), 1)
    d = np.linalg.norm(P[..., IU[0], :] - P[..., IU[1], :], axis=-1)
    d = d * (alive[..., IU[0]] * alive[..., IU[1]])
    return np.concatenate([zone, zone[..., [0]], zone[..., [6]], d], -1).astype(np.float32)


def context_vec(ctx, snap_s):
    n = len(ctx)
    o = np.zeros((n, len(OBJ)))
    o[np.arange(n), [OBJ.index(x) for x in ctx["objective"]]] = 1
    base = np.column_stack([np.clip(ctx["gold_diff"].to_numpy(float) / 5000, -3, 3), (ctx["side"].to_numpy() == 100) * 2 - 1,
                            ctx["spawn_s"].to_numpy(float) / 1800, o])
    return np.repeat(base[:, None, :], len(snap_s), 1).astype(np.float32)


def kselect(Z, seed):
    rows = []
    for k in range(2, 16):
        lab = KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
        rows.append({"K": k, "silhouette": silhouette_score(Z, lab), "dbi": davies_bouldin_score(Z, lab),
                     "ch": calinski_harabasz_score(Z, lab)})
    df = pd.DataFrame(rows)
    df["mean_rank"] = (rankdata(-df["silhouette"], method="dense") + rankdata(df["dbi"], method="dense")
                       + rankdata(-df["ch"], method="dense")) / 3
    return df


def cramers(a, b):
    tab = pd.crosstab(a, b)
    c2, p, *_ = chi2_contingency(tab)
    return float(c2), float(p), float(np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1))))


def cmd_fit(variant, seed):
    X, P, M, ctx = vdata.load()
    FIT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_va, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(X, groups=ctx["MatchId"]))
    tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed + 1).split(X[tr_va], groups=ctx["MatchId"].to_numpy()[tr_va]))
    tr, va = tr_va[tr], tr_va[va]
    Tn = X.shape[1]
    Xt, Pt, Yt = torch.tensor(X), torch.tensor(P), torch.tensor(targets(X, P))
    enc = Encoder(X.shape[-1])
    opt = torch.optim.AdamW(enc.parameters(), 1e-3, weight_decay=1e-4)
    w = torch.tensor([1.0] * 7 + [0.5] * 2 + [1.0] * 10)

    def loss(idx):
        x, p, y = Xt[idx].flatten(0, 1), Pt[idx].flatten(0, 1), Yt[idx].flatten(0, 1)
        return (((enc(x, p)[1] - y) ** 2) * w).mean()

    best, bad = 1e9, 0
    state = {k: v.clone() for k, v in enc.state_dict().items()}
    for ep in range(40):
        enc.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            opt.zero_grad()
            loss(perm[i:i + 128]).backward()
            opt.step()
        enc.eval()
        with torch.no_grad():
            v = float(loss(va))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: x.clone() for k, x in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    enc.load_state_dict(state)
    enc.eval()
    with torch.no_grad():
        test_loss = float(loss(te))
        E = enc.embed(Xt.flatten(0, 1), Pt.flatten(0, 1)).reshape(len(X), Tn, -1)
        g = torch.Generator().manual_seed(1000 + seed)                       # fixed random context projection + GRU
        proj = torch.nn.Linear(3 + len(OBJ), 32)
        gru = torch.nn.GRU(96, 64, batch_first=True)
        for m in (proj, gru):
            for p_ in m.parameters():
                p_.copy_(torch.randn(p_.shape, generator=g) * 0.2)
        c = torch.tensor(context_vec(ctx, vdata.PRE))
        c = proj(c) if variant == "ctx" else torch.zeros(len(X), Tn, 32)
        Z = gru(torch.cat([E, c], -1))[0][:, -1].numpy()
    lab = KMeans(K_TAG, n_init=20, random_state=seed).fit(Z)
    ksel = kselect(Z, seed)
    np.savez_compressed(FIT / f"tagnet_{variant}_seed{seed}.npz", Z=Z, labels=lab.labels_, centers=lab.cluster_centers_,
                        test_loss=test_loss, val_loss=best, epochs=ep + 1, tr=tr, te=te)
    ksel.to_csv(FIT / f"tagnet_{variant}_seed{seed}_ksel.csv", index=False)
    ctx.to_parquet(FIT / "tagnet_setups.parquet", index=False)
    print(variant, seed, "test recon", round(test_loss, 5), "epochs", ep + 1, "n", len(X), "sizes", np.bincount(lab.labels_))


def counts_features(X):
    alive = X[..., 7:8]
    return (X[..., :7] * alive).sum(2).reshape(len(X), -1)


def cmd_aggregate():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = pd.read_parquet(FIT / "tagnet_setups.parquet")
    X, P, M, ctx2 = vdata.load()
    ctx["gold_bin"] = pd.qcut(ctx["gold_diff"], 3, labels=["trail", "even", "lead"])
    rows, labs = [], {}
    for v in ("ctx", "noctx"):
        fs = sorted(FIT.glob(f"tagnet_{v}_seed*.npz"), key=lambda f: int(f.stem.split("seed")[1]))
        L = [np.load(f)["labels"] for f in fs]
        labs[v] = L
        pair = [adjusted_rand_score(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]
        for f, lab in zip(fs, L):
            z = np.load(f)
            cg, cs, co = cramers(lab, ctx["gold_bin"]), cramers(lab, ctx["side"]), cramers(lab, ctx["objective"])
            rows.append({"variant": v, "seed": int(f.stem.split("seed")[1]), "test_recon": float(z["test_loss"]),
                         "min_cluster": int(np.bincount(lab).min()), "V_gold": cg[2], "p_gold": cg[1], "V_side": cs[2],
                         "V_objective": co[2], "pairwise_ari_mean": float(np.mean(pair)) if pair else np.nan})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "tagnet_per_seed.csv", index=False)
    ks = pd.concat([pd.read_csv(f).assign(variant=f.stem.split("_")[1]) for f in FIT.glob("tagnet_*_ksel.csv")])
    ks.groupby(["variant", "K"])[["silhouette", "dbi", "ch", "mean_rank"]].mean().reset_index().to_csv(
        OUT / "tagnet_k_selection.csv", index=False)
    C = counts_features(X)
    cl = KMeans(K_TAG, n_init=20, random_state=0).fit_predict(C)
    out = {"n_setups": len(X), "count_baseline_vs_tagnet_ari": float(np.mean([adjusted_rand_score(cl, lab) for lab in labs["noctx"]])),
           "count_baseline_silhouette": float(silhouette_score(C, cl))}
    for v in ("ctx", "noctx"):
        sub = per[per["variant"] == v]
        out[f"{v}_pairwise_ari"] = float(sub["pairwise_ari_mean"].mean())
        out[f"{v}_V_gold_mean"] = float(sub["V_gold"].mean())
        out[f"{v}_V_gold_sd"] = float(sub["V_gold"].std())
        out[f"{v}_test_recon"] = float(sub["test_recon"].mean())
    out["leakage_inflation_V_gold"] = out["ctx_V_gold_mean"] / max(out["noctx_V_gold_mean"], 1e-9)
    out["ctx_vs_noctx_ari"] = float(np.mean([adjusted_rand_score(a, b) for a, b in zip(labs["ctx"], labs["noctx"])]))
    # series-bootstrap stability of the K-Means step (seed 0 embeddings): refit on a series resample, assign all, ARI vs full fit
    rng = np.random.default_rng(0)
    series_of = ctx["MatchId"].to_numpy()
    for v in ("noctx", "ctx"):
        z0 = np.load(FIT / f"tagnet_{v}_seed0.npz")
        Z, full = z0["Z"], z0["labels"]
        series = np.unique(series_of)
        aris = []
        for b in range(200):
            take = rng.choice(series, len(series))
            idx = np.concatenate([np.flatnonzero(series_of == s) for s in take])
            km = KMeans(K_TAG, n_init=5, random_state=b).fit(Z[idx])
            aris.append(adjusted_rand_score(full, km.predict(Z)))
        out[f"{v}_bootstrap_ari_mean"] = float(np.mean(aris))
        out[f"{v}_bootstrap_ari_q05"] = float(np.quantile(aris, 0.05))
        np.save(OUT / f"tagnet_{v}_bootstrap_ari.npy", np.array(aris))
    (OUT / "tagnet_summary.json").write_text(json.dumps(out, indent=1))
    asg = ctx.copy()
    asg["tagnet_noctx"], asg["tagnet_ctx"], asg["count_cluster"] = labs["noctx"][0], labs["ctx"][0], cl
    asg.to_parquet(OUT / "assignments.parquet", index=False)
    print(json.dumps(out, indent=1))
    print(per.groupby("variant").mean(numeric_only=True).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "aggregate"])
    ap.add_argument("--variant", default="noctx")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_fit(a.variant, a.seed) if a.cmd == "fit" else cmd_aggregate()
