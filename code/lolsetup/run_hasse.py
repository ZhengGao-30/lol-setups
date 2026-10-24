"""Fit HASSE for one (K, fold); aggregate across runs.

    python -m lolsetup.run_hasse fit --K 4 --fold -1   # full data fit (+ restarts)
    python -m lolsetup.run_hasse fit --K 4 --fold 2    # series-grouped CV: train on other folds, score fold 2
    python -m lolsetup.run_hasse aggregate             # choose K (held-out log-likelihood, ICL) and summarise
    add --layer v for the V layer (minimap tracks: every spawn, 13 snapshots from s-120 to s+60 every 15 s)

Layers: r = exact V5 frames at s-60, s, s+60 of the fixed-time first spawns (one setup per game x objective x
side); v = setups_v (one setup per game x objective x spawn index x side; snapshots without a live track frame
are missing observations). Objective classes (design strata): 0 = early river (dragon, grubs, herald),
1 = Atakhan, 2 = baron and Elder.
Outputs: fits/hasse{_v}/K{K}_fold{f}.json  and  results/hasse{_v}/selection.csv, best_patterns.json, posteriors.parquet
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import geometry, hasse, io

CLASS = {"DRAGON": 0, "HORDE": 0, "RIFTHERALD": 0, "ATAKHAN": 1, "BARON_NASHOR": 2, "ELDER": 2}
LAYERS = {"r": ("r_layer_setups.parquet", (-60, 0, 60), ["RiotPlatformGameId", "objective", "side"], ""),
          "v": ("v_layer_setups.parquet", tuple(range(-120, 61, 15)),
                ["RiotPlatformGameId", "objective", "idx", "side"], "_v"),
          # robustness layers: 7-snapshot grid s-90..s; formation radius 1500 / 2500 instead of 1900
          "v7": ("v_layer_setups.parquet", tuple(range(-90, 1, 15)), ["RiotPlatformGameId", "objective", "idx", "side"], "_v7"),
          "vr1500": ("v_layer_setups_r1500.parquet", tuple(range(-120, 61, 15)), ["RiotPlatformGameId", "objective", "idx", "side"], "_vr1500"),
          "vr2500": ("v_layer_setups_r2500.parquet", tuple(range(-120, 61, 15)), ["RiotPlatformGameId", "objective", "idx", "side"], "_vr2500")}
SRC, OFFSETS, KEY, TAG = LAYERS["r"]
RESTARTS = 10
SEED0 = 0
PART = 0  # restarts are split over tasks: part p uses seeds p * RESTARTS .. (p + 1) * RESTARTS - 1
MONO = False  # H_mono: centre paths restricted to merges only
CTXCLASS = False  # positive control: context (gold-difference tercile) enters the mixture weights = TAGNet-style leakage
FOLDS = 5


def set_layer(layer, ctxclass=False, mono=False):
    global SRC, OFFSETS, KEY, TAG, CTXCLASS, MONO
    SRC, OFFSETS, KEY, TAG = LAYERS[layer]
    CTXCLASS, MONO = ctxclass, mono
    if ctxclass:
        TAG += "ctx"
    if mono:
        TAG += "mono"


def n_classes():
    return 9 if CTXCLASS else 3


def load_setups():
    d = pd.read_parquet(io.ROOT / "fits" / SRC)
    games = io.core_games()[["GameId", "MatchId"]]
    d = d.merge(games, on="GameId", how="left").sort_values("offset_s")
    zone_idx = {z: i for i, z in enumerate(geometry.ZONES)}
    span = OFFSETS[-1] - OFFSETS[0]
    data, keys = [], []
    if CTXCLASS:
        gd0 = d[d["offset_s"] == 0].groupby(KEY)["gold_diff"].first()
        terc = pd.qcut(gd0, 3, labels=False).to_dict()
    for key, g in d.groupby(KEY, sort=True):
        first = g.iloc[0]
        g = g.set_index("offset_s").reindex(list(OFFSETS))
        snaps = []
        for off, r in g.iterrows():
            s_cov = (off - OFFSETS[0]) / span
            if pd.isna(r["partition"]) or (off > 0 and not r["obj_up"]):
                snaps.append({"missing": True, "s": s_cov})  # no live frame, or risk set: objective already taken
                continue
            A = tuple(i for i, ch in enumerate(io.ROLE_SHORT) if r[f"alive_{ch}"])
            blocks = tuple(tuple(io.ROLE_SHORT.index(ch) for ch in blk) for blk in r["partition"].split("|") if blk)
            bz = [zone_idx[z] for z in r["block_zones"].split("|")] if blocks else []
            snaps.append({"A": A, "partition": blocks, "block_zones": bz, "s": s_cov})
        cls = CLASS[first["objective"]] * 3 + int(terc.get(key, 1)) if CTXCLASS else CLASS[first["objective"]]
        data.append(hasse.make_setup(snaps, cls, str(first["MatchId"]),
                                     {"team": first["team"], "opponent": first["opponent"], "event": first["event"]}))
        keys.append(key)
    return data, keys


def one_restart(data, B, K, s):
    """Even seeds: k-medoids centre paths on the lattice path distance; odd seeds: random centre paths."""
    init = hasse.medoid_init(data, K, seed=s) if s % 2 == 0 and K > 1 else None
    return hasse.Hasse(K=K, n_classes=n_classes(), T=len(OFFSETS), seed=s, init_mu=None if init is None else init[0], mono=MONO).fit(B, max_iter=300)


def fit_best(data, K, restarts=None, seed0=0):
    B = hasse.Batch(data)
    restarts = RESTARTS if restarts is None else restarts
    best = None
    for s in range(restarts):
        m = one_restart(data, B, K, seed0 + s)
        if best is None or m.loglik_ > best.loglik_:
            best = m
    return best, B


def cmd_fit(K, fold):
    data, keys = load_setups()
    out = io.ROOT / "fits" / f"hasse{TAG}"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_p{PART}" if PART else ""
    res = {"K": K, "fold": fold, "part": PART, "n_setups": len(data)}
    if fold < 0:
        B = hasse.Batch(data)
        runs = []
        best = None
        for s in range(RESTARTS):
            m = one_restart(data, B, K, SEED0 + s)
            runs.append(m.loglik_)
            if best is None or m.loglik_ > best.loglik_:
                best = m
        n_obs = B.N
        p = best.n_params()
        ent = -np.sum(best.gamma_[:, 0] * np.log(np.maximum(best.gamma_[:, 0], 1e-300)))
        res.update(loglik=best.loglik_, restart_logliks=runs, n_params=p,
                   bic=-2 * best.loglik_ + p * np.log(len(data)), icl=-2 * best.loglik_ + p * np.log(len(data)) + 2 * ent,
                   patterns=best.summary(), xi=best.xi)
        with open(out / f"K{K}_full{suffix}.pkl", "wb") as f:
            pickle.dump({"model": best, "keys": keys}, f)
    else:
        groups = np.array([e.group for e in data])
        tr, te = list(GroupKFold(n_splits=FOLDS).split(np.zeros(len(data)), groups=groups))[fold]
        m, _ = fit_best([data[i] for i in tr], K, seed0=SEED0)
        res.update(train_loglik=m.loglik_, test_loglik=float(m.loglik_per_setup([data[i] for i in te]).sum()),
                   n_test=len(te))
    with open(out / f"K{K}_fold{fold}{suffix}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "patterns"}, default=float))


def labels_of(model, data):
    """Hard pattern label of every setup: argmax of its mean posterior occupancy over the snapshots."""
    B = hasse.Batch(data)
    _, gamma, _ = model.forward_backward(B)
    return gamma.mean(1).argmax(1)


def cmd_boot(K, batch, restarts):
    """Series-bootstrap stability: refit on a resample of series, label every setup, ARI against the full-data labels."""
    from sklearn.metrics import adjusted_rand_score
    data, keys = load_setups()
    src = io.ROOT / "fits" / f"hasse{TAG}"
    best_pkl = sorted(src.glob(f"K{K}_full*.pkl"))
    full = max((pickle.load(open(f, "rb"))["model"] for f in best_pkl), key=lambda m: m.loglik_)
    lab_full = labels_of(full, data)
    groups = np.array([e.group for e in data])
    uniq = np.unique(groups)
    rng = np.random.default_rng(5000 + batch)
    take = rng.choice(uniq, len(uniq))
    idx = np.concatenate([np.flatnonzero(groups == g) for g in take])
    m, _ = fit_best([data[i] for i in idx], K, restarts=restarts, seed0=100 * (batch + 1))
    lab = labels_of(m, data)
    out = {"K": K, "batch": batch, "ari_vs_full": float(adjusted_rand_score(lab_full, lab)), "loglik": m.loglik_,
           "sizes_full": np.bincount(lab_full, minlength=K).tolist(), "sizes_boot": np.bincount(lab, minlength=K).tolist()}
    (src / f"boot_K{K}_b{batch}.json").write_text(json.dumps(out))
    print(json.dumps(out))


def cmd_stability(K):
    """Cross-restart-batch ARI of the full-data solutions (analogue of TAGNet's cross-seed ARI) and bootstrap summary."""
    from sklearn.metrics import adjusted_rand_score
    data, keys = load_setups()
    src = io.ROOT / "fits" / f"hasse{TAG}"
    out = io.RESULTS / f"hasse{TAG}"
    out.mkdir(parents=True, exist_ok=True)
    labs = [labels_of(pickle.load(open(f, "rb"))["model"], data) for f in sorted(src.glob(f"K{K}_full*.pkl"))]
    pair = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    boots = [json.loads(f.read_text())["ari_vs_full"] for f in src.glob(f"boot_K{K}_b*.json")]
    res = {"K": K, "n_solutions": len(labs), "cross_restart_ari_mean": float(np.mean(pair)) if pair else None,
           "cross_restart_ari_min": float(np.min(pair)) if pair else None,
           "bootstrap_n": len(boots), "bootstrap_ari_mean": float(np.mean(boots)) if boots else None,
           "bootstrap_ari_q05": float(np.quantile(boots, 0.05)) if boots else None}
    (out / f"stability_K{K}.json").write_text(json.dumps(res, indent=1))
    np.save(out / f"bootstrap_ari_K{K}.npy", np.array(boots))
    print(json.dumps(res, indent=1))


def cmd_mono_compare(K):
    """H_mono: held-out log-likelihood of the merge-only (gather) restriction against the unrestricted fit, fold by fold."""
    def best(dirn):
        out = {}
        for f in (io.ROOT / "fits" / dirn).glob(f"K{K}_fold*.json"):
            r = json.loads(f.read_text())
            if r["fold"] >= 0 and (r["fold"] not in out or r["train_loglik"] > out[r["fold"]]["train_loglik"]):
                out[r["fold"]] = r
            if r["fold"] < 0 and "full" not in out or (r["fold"] < 0 and r["loglik"] > out["full"]["loglik"]):
                out["full"] = r
        return out
    a, b = best(f"hasse{TAG.replace('mono', '')}"), best(f"hasse{TAG}")
    rows = [{"fold": f, "test_ll_unrestricted": a[f]["test_loglik"], "test_ll_mono": b[f]["test_loglik"], "n_test": a[f]["n_test"]}
            for f in sorted(k for k in b if k != "full") if f in a]
    df = pd.DataFrame(rows)
    df["diff_per_setup"] = (df["test_ll_unrestricted"] - df["test_ll_mono"]) / df["n_test"]
    full = {"K": K, "loglik_unrestricted": a["full"]["loglik"], "loglik_mono": b["full"]["loglik"],
            "heldout_diff_per_setup_mean": float(df["diff_per_setup"].mean()), "heldout_diff_per_setup_sd_over_folds": float(df["diff_per_setup"].std()),
            "folds_favouring_unrestricted": int((df["diff_per_setup"] > 0).sum()), "n_folds": len(df)}
    out = io.RESULTS / f"hasse{TAG}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"mono_compare_K{K}.csv", index=False)
    (out / f"mono_compare_K{K}.json").write_text(json.dumps(full, indent=1))
    print(df.to_string())
    print(json.dumps(full, indent=1))


def cmd_aggregate(k_use=None):
    src = io.ROOT / "fits" / f"hasse{TAG}"
    rows = []
    for p in sorted(src.glob("K*_fold*.json")):
        rows.append(json.loads(p.read_text()))
    df = pd.DataFrame(rows)
    if "part" not in df:
        df["part"] = 0
    df["part"] = df["part"].fillna(0).astype(int)
    df["score"] = df["loglik"].fillna(df.get("train_loglik"))            # best restart batch per (K, fold) by training likelihood
    df = df.sort_values("score", ascending=False).drop_duplicates(["K", "fold"]).drop(columns="score")
    best_part = df[df["fold"] < 0].set_index("K")["part"].to_dict()
    full = df[df["fold"] < 0].set_index("K")[["loglik", "bic", "icl", "n_params"]]
    cv = df[df["fold"] >= 0].groupby("K").agg(test_loglik=("test_loglik", "sum"), folds=("fold", "nunique"))
    sel = full.join(cv)
    sel["test_loglik_per_setup"] = sel["test_loglik"] / df[df["fold"] >= 0].groupby("K")["n_test"].sum()
    out = io.RESULTS / f"hasse{TAG}"
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "selection.csv")
    best_k = int(sel["test_loglik"].idxmax())
    k_use = k_use or best_k   # patterns / posteriors are written for k_use (default: best held-out K)
    sel["bic_rank"] = sel["bic"].rank(); sel["icl_rank"] = sel["icl"].rank()
    sel.to_csv(out / "selection.csv")
    with open(src / f"K{k_use}_full{'_p' + str(best_part[k_use]) if best_part[k_use] else ''}.pkl", "rb") as f:
        obj = pickle.load(f)
    m, keys = obj["model"], obj["keys"]
    (out / "best_patterns.json").write_text(json.dumps({"K": k_use, "patterns": m.summary(), "xi": m.xi}, indent=1))
    (out / f"patterns_K{k_use}.json").write_text(json.dumps({"K": k_use, "patterns": m.summary(), "xi": m.xi, "q": m.q.tolist()}, indent=1))
    post = pd.DataFrame(m.gamma_[:, 0], columns=[f"p_pattern{c}" for c in range(m.K)])
    post[KEY[:len(keys[0])]] = pd.DataFrame(keys)
    for c in range(m.K):
        post[f"occ_pattern{c}"] = m.gamma_[:, :, c].mean(1)          # share of the setup spent in pattern c
        post[f"s_pattern{c}"] = m.gamma_[:, OFFSETS.index(0), c]    # posterior at the spawn snapshot
    post.to_parquet(out / "posteriors.parquet", index=False)
    post.to_parquet(out / f"posteriors_K{k_use}.parquet", index=False)
    np.savez_compressed(out / "gamma.npz", gamma=m.gamma_, mu=m.mu, theta=m.theta, iota=m.iota, xi=m.xi, q=m.q, w=m.w, a=m.a, b=m.b)
    print(sel.to_string())
    print("best K by held-out log-likelihood:", best_k, "| K by BIC:", int(sel["bic"].idxmin()), "| K by ICL:", int(sel["icl"].idxmin()), "| written for K =", k_use)
    for s in m.summary():
        print(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "aggregate", "boot", "stability", "mono_compare"])
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--K", type=int, default=0, help="fit: number of patterns; aggregate: K to write posteriors for (<= 0: best held-out)")
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--layer", choices=list(LAYERS), default="r")
    ap.add_argument("--part", type=int, default=0, help="restart batch (parallel tasks use different parts)")
    ap.add_argument("--seed0", type=int, default=0, help="first restart seed (split restarts across tasks)")
    ap.add_argument("--ctxclass", action="store_true", help="positive control: gold tercile enters the class weights")
    ap.add_argument("--mono", action="store_true", help="H_mono: centre paths may only merge")
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    a = ap.parse_args()
    set_layer(a.layer, a.ctxclass, a.mono)
    RESTARTS = a.restarts
    PART = a.part
    SEED0 = a.part * a.restarts + a.seed0
    if a.cmd == "fit":
        cmd_fit(a.K, a.fold)
    elif a.cmd == "boot":
        cmd_boot(a.K, a.batch, a.restarts)
    elif a.cmd == "stability":
        cmd_stability(a.K)
    elif a.cmd == "mono_compare":
        cmd_mono_compare(a.K)
    else:
        cmd_aggregate(a.K if a.K > 0 else None)
