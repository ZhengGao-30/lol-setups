"""Team fingerprints: HASSE pattern usage (occupancy shares) plus LB-QRE parameters per team, with
series-block bootstrap intervals; nearest neighbours, split-half reliability and the T1 position.

    python -m lolsetup.fingerprints --tag _v
Outputs: results/fingerprints/{usage,usage_ci,neighbours,split_half,theta}.csv
"""
import argparse
import pickle

import numpy as np
import pandas as pd

from . import io
from .context import KEY

OUT = io.RESULTS / "fingerprints"
MIN_SETUPS = 30


def usage(d, occ):
    return d.groupby("team")[occ].mean()


def main(tag="_v", boot=1000, k=None):
    global OUT
    OUT = io.RESULTS / ("fingerprints" + (f"_K{k}" if k else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    cov = pd.read_parquet(io.RESULTS / "context" / "context_covariates.parquet")
    post = pd.read_parquet(io.RESULTS / f"hasse{tag}" / (f"posteriors_K{k}.parquet" if k else "posteriors.parquet"))
    d = cov.merge(post, on=KEY, how="inner")
    occ = [c for c in d.columns if c.startswith("occ_pattern")]
    n = d.groupby("team").size()
    keep = n[n >= MIN_SETUPS].index
    d = d[d["team"].isin(keep)]
    U = usage(d, occ)
    U.assign(n_setups=n[U.index]).to_csv(OUT / "usage.csv")
    rng = np.random.default_rng(0)
    series = d["MatchId"].to_numpy()
    uniq = np.unique(series)
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    teams = list(U.index)
    boots = []
    for b in range(boot):
        take = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx_by[s] for s in take])
        ub = usage(d.iloc[ii], occ).reindex(teams)
        boots.append(ub.to_numpy())
    boots = np.array(boots)                               # B x teams x K
    rows = []
    for i, t in enumerate(teams):
        for c, name in enumerate(occ):
            v = boots[:, i, c]
            v = v[~np.isnan(v)]
            rows.append({"team": t, "pattern": c, "usage": U.iloc[i, c], "lo": np.quantile(v, 0.025), "hi": np.quantile(v, 0.975),
                         "n_setups": int(n[t])})
    pd.DataFrame(rows).to_csv(OUT / "usage_ci.csv", index=False)
    # rank probabilities of T1 per pattern and nearest-neighbour distance (total variation) with intervals
    tv = lambda a, b: 0.5 * np.abs(a - b).sum()
    if "T1" in teams:
        i1 = teams.index("T1")
        nn = []
        for b in range(boot):
            ub = boots[b]
            ok = ~np.isnan(ub).any(1)
            dist = np.array([tv(ub[i1], ub[j]) if ok[j] and j != i1 else np.inf for j in range(len(teams))])
            nn.append(np.argmin(dist))
        share = pd.Series(np.bincount(nn, minlength=len(teams)) / boot, index=teams, name="p_nearest_to_T1").sort_values(ascending=False)
        D = pd.Series({t: tv(U.loc["T1"].to_numpy(), U.loc[t].to_numpy()) for t in teams if t != "T1"}, name="tv_to_T1")
        pd.concat([D, share], axis=1).sort_values("tv_to_T1").to_csv(OUT / "neighbours_T1.csv")
        mean_other = np.array([tv(U.loc[t].to_numpy(), U.drop(index=t).mean().to_numpy()) for t in teams])
        pd.Series(mean_other, index=teams, name="tv_to_league_mean").to_csv(OUT / "distance_to_league.csv")
    # split-half reliability: random halves of each team's series, correlation of usage across teams, per pattern
    rel = []
    for r in range(200):
        a, bb = [], []
        for t in teams:
            if d.loc[d["team"] == t, "MatchId"].nunique() < 4:   # a half needs at least two series
                continue
            ser = np.array(d.loc[d["team"] == t, "MatchId"].unique(), dtype=object)
            rng.shuffle(ser)
            half = set(ser[: len(ser) // 2])
            m = d["team"] == t
            a.append(d.loc[m & d["MatchId"].isin(half), occ].mean().to_numpy())
            bb.append(d.loc[m & ~d["MatchId"].isin(half), occ].mean().to_numpy())
        a, bb = np.array(a), np.array(bb)
        rel.append([np.corrcoef(a[:, c], bb[:, c])[0, 1] for c in range(len(occ))])
    rel = np.array(rel)
    sh = pd.DataFrame({"pattern": range(len(occ)), "corr_mean": np.nanmean(rel, 0), "corr_lo": np.nanquantile(rel, 0.025, 0),
                       "corr_hi": np.nanquantile(rel, 0.975, 0)})
    sh["spearman_brown"] = 2 * sh["corr_mean"] / (1 + sh["corr_mean"])
    sh.to_csv(OUT / "split_half.csv", index=False)
    qf = io.ROOT / "fits" / "qre_fit.pkl"
    if qf.exists():
        q = pickle.load(open(qf, "rb"))
        th = pd.DataFrame(q["theta"], columns=["lam", "nu", "kappa", "psi", "omega_k", "omega_m"][:q["theta"].shape[1]], index=q["teams"])
        th["lam_obj"] = th["lam"] * th["nu"]
        th.to_csv(OUT / "theta.csv")
    print(U.round(3).to_string())
    print(sh.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_v")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    main(a.tag, a.boot, a.k)
