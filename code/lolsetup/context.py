"""Context stage (cut): HASSE pattern posteriors regressed on context, never fed back to the pattern layer.

    python -m lolsetup.context --tag _v        # uses results/hasse_v/posteriors.parquet (+ _vctx for the leakage control)

Soft-label multinomial logit: every setup contributes K weighted rows (weight = its occupancy share of pattern c),
the maximum-likelihood fit of a multinomial logit with fractional counts. Covariates: gold difference (per 1000 g,
at the spawn), blue side, time (min), own / opponent soul point (3 drakes), pre-match strength difference
(Elo per 100 points, computed from all earlier games in games_all), objective-class dummies. Uncertainty is a
series-block bootstrap (resample series, refit); effects are reported as coefficients against the reference
pattern and as average marginal effects on pattern occupancy. The positive control compares the association of the
pattern assignment with gold-difference terciles when gold enters the class weights (TAGNet-style leakage,
run_hasse --ctxclass) with the cut version.
Outputs: results/context/{coefficients,ame,leakage_control}.csv, context_covariates.parquet.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

from . import io

OUT = io.RESULTS / "context"
KEY = ["RiotPlatformGameId", "objective", "idx", "side"]
CLASS = {"DRAGON": 0, "HORDE": 0, "RIFTHERALD": 0, "ATAKHAN": 1, "BARON_NASHOR": 2, "ELDER": 2}


def elo_table(k=24.0, start=1500.0):
    """Pre-game Elo of both teams from all played games before it (sequential by date)."""
    g = io.played_games().sort_values("DateTime_UTC")
    r, rows = {}, []
    for x in g.itertuples():
        b, rd = r.get(x.Blue, start), r.get(x.Red, start)
        rows.append({"RiotPlatformGameId": x.RiotPlatformGameId, "elo_blue": b, "elo_red": rd})
        pb = 1 / (1 + 10 ** ((rd - b) / 400))
        res = 1.0 if x.WinTeam == x.Blue else 0.0
        r[x.Blue] = b + k * (res - pb)
        r[x.Red] = rd - k * (res - pb)
    return pd.DataFrame(rows)


def covariates(setups_path=None):
    v = pd.read_parquet(setups_path or io.ROOT / "fits" / "v_layer_setups.parquet")
    games = io.core_games()[["GameId", "MatchId"]]
    at0 = v[v["offset_s"] == 0].drop_duplicates(KEY).merge(games, on="GameId", how="left")
    fs = pd.read_parquet(io.ROOT / "fits" / "frames_state.parquet")
    elo = elo_table()
    rows = []
    fs_by = {g: d.sort_values("frame_ms") for g, d in fs.groupby("RiotPlatformGameId")}
    for r in at0.itertuples():
        fg = fs_by.get(r.RiotPlatformGameId)
        j = int(np.searchsorted(fg["frame_ms"].to_numpy() / 1000, r.spawn_s, side="right")) - 1 if fg is not None else -1
        own, opp = (("b_drakes", "r_drakes") if r.side == 100 else ("r_drakes", "b_drakes"))
        rows.append({"RiotPlatformGameId": r.RiotPlatformGameId, "objective": r.objective, "idx": r.idx, "side": r.side,
                     "MatchId": str(r.MatchId), "team": r.team, "opponent": r.opponent, "event": r.event,
                     "gold_k": r.gold_diff / 1000, "blue": float(r.side == 100), "minute": r.spawn_s / 60,
                     "own_soul_pt": float(fg[own].iloc[j] == 3) if j >= 0 else np.nan,
                     "opp_soul_pt": float(fg[opp].iloc[j] == 3) if j >= 0 else np.nan, "cls": CLASS[r.objective]})
    c = pd.DataFrame(rows).merge(elo, on="RiotPlatformGameId", how="left")
    c["elo_diff"] = np.where(c["side"] == 100, c["elo_blue"] - c["elo_red"], c["elo_red"] - c["elo_blue"]) / 100
    return c.drop(columns=["elo_blue", "elo_red"])


FEATS = ["gold_k", "blue", "minute", "own_soul_pt", "opp_soul_pt", "elo_diff", "cls1", "cls2"]


def design(d):
    X = d[["gold_k", "blue", "minute", "own_soul_pt", "opp_soul_pt", "elo_diff"]].copy()
    X["cls1"], X["cls2"] = (d["cls"] == 1).astype(float), (d["cls"] == 2).astype(float)
    return X


def fit_soft(X, W, C=1e4):
    """Multinomial logit with fractional counts: W (n x K) row weights. Returns coef (K x p), intercept (K,)."""
    n, K = W.shape
    Xs = np.tile(X.to_numpy(float), (K, 1))
    y = np.repeat(np.arange(K), n)
    w = W.T.reshape(-1)
    m = LogisticRegression(C=C, max_iter=2000).fit(Xs, y, sample_weight=np.maximum(w, 1e-9))
    return m, m.coef_, m.intercept_


def ame(m, X, var):
    """Average marginal effect on pattern occupancy of +1 unit in `var`."""
    Xp = X.copy()
    Xp[var] = Xp[var] + 1
    return (m.predict_proba(Xp.to_numpy(float)) - m.predict_proba(X.to_numpy(float))).mean(0)


def cramers(a, b):
    tab = pd.crosstab(a, b)
    c2, p, *_ = chi2_contingency(tab)
    return float(np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1)))), float(p)


def main(tag="_v", boot=500, k=None):
    OUT.mkdir(parents=True, exist_ok=True)
    cov = covariates()
    cov.to_parquet(OUT / "context_covariates.parquet", index=False)
    sfx = f"_K{k}" if k else ""
    post = pd.read_parquet(io.RESULTS / f"hasse{tag}" / (f"posteriors_K{k}.parquet" if k else "posteriors.parquet"))
    d = cov.merge(post, on=KEY, how="inner").dropna(subset=["gold_k", "elo_diff", "own_soul_pt"])
    occ = [c for c in d.columns if c.startswith("occ_pattern")]
    K = len(occ)
    W = d[occ].to_numpy()
    W = W / W.sum(1, keepdims=True)
    X = design(d)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Xz = (X - mu) / sd
    m, coef, _ = fit_soft(Xz, W)
    # report against the last pattern as reference
    ref = coef - coef[-1]
    series = d["MatchId"].to_numpy()
    uniq = np.unique(series)
    rng = np.random.default_rng(0)
    B_coef, B_ame = [], []
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    for b in range(boot):
        take = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx_by[s] for s in take])
        mb, cb, _ = fit_soft(Xz.iloc[ii], W[ii])
        B_coef.append(cb - cb[-1])
        B_ame.append(np.stack([ame(mb, Xz.iloc[ii], v) for v in X.columns], 1))
    B_coef, B_ame = np.array(B_coef), np.array(B_ame)
    rows = []
    for c in range(K - 1):
        for j, f in enumerate(X.columns):
            lo, hi = np.quantile(B_coef[:, c, j], [0.025, 0.975])
            rows.append({"pattern": c, "ref": K - 1, "feature": f, "coef_per_sd": ref[c, j], "lo": lo, "hi": hi,
                         "sd": sd[f], "sig": bool(lo > 0 or hi < 0)})
    pd.DataFrame(rows).to_csv(OUT / f"coefficients{tag}{sfx}.csv", index=False)
    pt = np.stack([ame(m, Xz, v) for v in X.columns], 1)
    rows = []
    for c in range(K):
        for j, f in enumerate(X.columns):
            lo, hi = np.quantile(B_ame[:, c, j], [0.025, 0.975])
            rows.append({"pattern": c, "feature": f, "ame_per_sd": pt[c, j], "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0)})
    pd.DataFrame(rows).to_csv(OUT / f"ame{tag}{sfx}.csv", index=False)
    # gold-difference association of the hard assignment (and the leakage control when available)
    lc = []
    terc = pd.qcut(d["gold_k"], 3, labels=["trail", "even", "lead"])
    for name, t in (("cut", tag), ("leaky (gold in class weights)", tag + "ctx")):
        f = io.RESULTS / f"hasse{t}" / (f"posteriors_K{k}.parquet" if k else "posteriors.parquet")
        if not f.exists():
            continue
        p2 = cov.merge(pd.read_parquet(f), on=KEY, how="inner").dropna(subset=["gold_k"])
        o2 = [c for c in p2.columns if c.startswith("occ_pattern")]
        lab = np.array(o2)[p2[o2].to_numpy().argmax(1)]
        t2 = pd.qcut(p2["gold_k"], 3, labels=["trail", "even", "lead"])
        v_, p_ = cramers(lab, t2)
        lc.append({"model": f"HASSE {name}", "n": len(p2), "cramers_V_gold": v_, "p": p_})
    a = io.RESULTS / "baseline" / "assignments.parquet"
    if a.exists():
        asg = pd.read_parquet(a)
        t3 = pd.qcut(asg["gold_diff"], 3, labels=["trail", "even", "lead"])
        for col, name in (("tagnet_noctx", "TAGNet without context"), ("tagnet_ctx", "TAGNet with context (their design)")):
            v_, p_ = cramers(asg[col], t3)
            lc.append({"model": name, "n": len(asg), "cramers_V_gold": v_, "p": p_})
    pd.DataFrame(lc).to_csv(OUT / f"leakage_control{sfx}.csv", index=False)
    print(pd.DataFrame(lc).to_string())
    print(pd.read_csv(OUT / f"ame{tag}{sfx}.csv").query("sig").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_v")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--k", type=int, default=None, help="use results/hasse<tag>/posteriors_K<k>.parquet")
    a = ap.parse_args()
    main(a.tag, a.boot, a.k)
