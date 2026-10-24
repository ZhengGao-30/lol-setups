"""Fight laws from the V layer: stochastic Lanchester (S7) and the Lanchester -> Tullock contest function.

S7, one row per kill with unequal local numbers (n_more > n_fewer, fights.py):
    logit P(victim on the larger side) = c + b_n * ln(n_more / n_fewer) + b_g * ln(gold_more / gold_fewer)
  Stochastic square law (kill rate of a side proportional to its own numbers): b_n = -1.
  Stochastic linear law (rate proportional to both sides' numbers): b_n = 0.
  Also the binned P(victim on larger side) against the square-law curve n_fewer / (n_more + n_fewer).
Tullock, one row per decisive fight (blue perspective; antisymmetric: swapping teams flips every regressor):
    logit P(blue wins) = gamma_n * ln(n_blue / n_red) + gamma_g * ln(gold_blue / gold_red) + gamma_side
  gamma_n / gamma_g = value of one extra player in units of log gold per player.
Uncertainty: series-block bootstrap (B resamples). Sensitivity: track-only counts instead of track + V5 participants.

Usage:  python -m lolsetup.analysis_fights -> results/fights/*.csv + summary.md
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import io
from .analysis_r import md

B = 1000
rng = np.random.default_rng(20261022)


def _logit(X, y, intercept=True):
    m = LogisticRegression(C=1e6, fit_intercept=intercept, max_iter=2000).fit(X, y)
    return np.r_[m.intercept_ if intercept else [], m.coef_[0]]


def _boot(df, fn):
    est = fn(df)
    pos = {k: np.asarray(v) for k, v in df.reset_index(drop=True).groupby("MatchId").indices.items()}
    keys, base = list(pos), df.reset_index(drop=True)
    draws = np.array([fn(base.iloc[np.concatenate([pos[k] for k in rng.choice(keys, len(keys))])]) for _ in range(B)])
    return est, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def s7_frame(d, tracked=False):
    nb, nr = (d["n_blue_tracked"], d["n_red_tracked"]) if tracked else (d["n_blue"], d["n_red"])
    d = d.assign(nb=nb, nr=nr)
    d = d[(d["nb"] != d["nr"]) & (d[["nb", "nr"]].min(axis=1) > 0)].copy()
    blue_more = d["nb"] > d["nr"]
    d["n_more"], d["n_fewer"] = np.where(blue_more, d["nb"], d["nr"]), np.where(blue_more, d["nr"], d["nb"])
    d["g_more"] = np.where(blue_more, d["gold_pp_blue"], d["gold_pp_red"])
    d["g_fewer"] = np.where(blue_more, d["gold_pp_red"], d["gold_pp_blue"])
    d["victim_larger"] = (d["victim_side"] == np.where(blue_more, 100, 200)).astype(int)
    d["ln_n"] = np.log(d["n_more"] / d["n_fewer"])
    d["ln_g"] = np.log(d["g_more"] / d["g_fewer"])
    return d.dropna(subset=["ln_g"])


def fight_frame(f, tracked=False):
    nb, nr = (f["n_blue_tracked"], f["n_red_tracked"]) if tracked else (f["n_blue"], f["n_red"])
    f = f.assign(nb=nb, nr=nr)
    f = f[f["winner"].notna() & (f["nb"] > 0) & (f["nr"] > 0)].copy()
    f["blue_won"] = (f["winner"] == 100).astype(int)
    f["ln_n"] = np.log(f["nb"] / f["nr"])
    f["ln_g"] = np.log(f["gold_pp_blue"] / f["gold_pp_red"])
    f["side"] = 1.0
    return f.dropna(subset=["ln_g"])


def main():
    out = io.RESULTS / "fights"
    out.mkdir(parents=True, exist_ok=True)
    games = io.core_games()[["RiotPlatformGameId", "MatchId"]]
    d = pd.read_parquet(io.ROOT / "fits" / "deaths_numbers.parquet").merge(games, on="RiotPlatformGameId")
    f = pd.read_parquet(io.ROOT / "fits" / "fights.parquet").merge(games, on="RiotPlatformGameId")
    lines = ["# Fight laws (V layer)", ""]
    rows = []
    for tracked in (False, True):
        s7 = s7_frame(d, tracked)
        e, lo, hi = _boot(s7, lambda x: _logit(x[["ln_n", "ln_g"]].to_numpy(), x["victim_larger"].to_numpy()))
        for name, i in (("c", 0), ("b_n (square law -1, linear 0)", 1), ("b_g", 2)):
            rows.append({"model": "S7 next death", "counts": "tracked only" if tracked else "tracked + V5 participants",
                         "param": name, "est": e[i], "lo": lo[i], "hi": hi[i], "n": len(s7)})
        fx = fight_frame(f, tracked)
        e, lo, hi = _boot(fx, lambda x: _logit(x[["ln_n", "ln_g", "side"]].to_numpy(), x["blue_won"].to_numpy(),
                                               intercept=False))
        ratio = _boot(fx, lambda x: (lambda b: b[0] / b[1])(_logit(x[["ln_n", "ln_g", "side"]].to_numpy(),
                                                                   x["blue_won"].to_numpy(), intercept=False)))
        for name, i in (("gamma_n", 0), ("gamma_g", 1), ("gamma_side", 2)):
            rows.append({"model": "Tullock fight", "counts": "tracked only" if tracked else "tracked + V5 participants",
                         "param": name, "est": e[i], "lo": lo[i], "hi": hi[i], "n": len(fx)})
        rows.append({"model": "Tullock fight", "counts": "tracked only" if tracked else "tracked + V5 participants",
                     "param": "gamma_n / gamma_g", "est": ratio[0], "lo": ratio[1], "hi": ratio[2], "n": len(fx)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "fight_laws.csv", index=False)
    s7 = s7_frame(d)
    binned = (s7.assign(ratio=s7["n_more"].astype(str) + "v" + s7["n_fewer"].astype(str))
                .groupby("ratio").agg(n=("victim_larger", "size"), p_victim_larger=("victim_larger", "mean"),
                                      square_law=("n_fewer", lambda v: float(np.mean(v / (v + s7.loc[v.index, "n_more"]))))))
    binned = binned[binned["n"] >= 30]
    binned.to_csv(out / "s7_binned.csv")
    lines += ["Series-block bootstrap 95% intervals.", "", md(res, index=False), "",
              "P(next death on the larger side) by local numbers (cells with n >= 30) vs the square-law value:", "",
              md(binned), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
