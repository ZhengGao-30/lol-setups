"""R-layer descriptive analysis (exact Riot frames at s-60, s, s+60 of the fixed-time first spawns).

Mirrors TAGNet's analyses with corrected statistics; everything is descriptive (no prediction):
  1. formation vocabulary (5 alive): shapes, labelled partitions (rarefied), role singleton in 4+1
  2. pit allocation k_R by gold state: P(k=0), P(k in {0,5}), mean, variance           (S1, S2, S4)
  3. commitment at s+60: P(self >= 3 at the pit), P(both >= 3)
  4. arrival priority at s-60 vs who takes the objective within C = 180 s (one row per spawn)
  5. estimator audit: TAGNet-style unweighted team mean vs pooled vs within-team contrast
Rules: post-spawn rows (s+60) use the RISK SET only (objective still alive at the snapshot); cells with
fewer than MIN_N team-setups are reported as not estimable. k is computed for R in {1000, 1500, 2000}
(primary 1500); a trailing-vs-leading contrast is a headline only if its sign agrees at all three R and
it survives BH-FDR. Gold state: GD at s-60 (before the window) / archive SD of GD at that minute.
Uncertainty: series-block bootstrap; contrasts are computed inside each resample (paired).

Usage:  python -m lolsetup.analysis_r  -> results/r_layer/*.csv + summary.md
"""
import numpy as np
import pandas as pd

from . import geometry, io

B = 1000
MIN_N = 30
C_HORIZON = 180
rng = np.random.default_rng(20261022)
OBJ_ORDER = ["DRAGON", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR"]
RADII = tuple(int(r) for r in geometry.PIT_RADII)
PRIMARY_R = 1500


def gold_sd_by_minute():
    """SD of team gold difference per game minute over the archive (on-grid frames, game-end frame dropped)."""
    rows = []
    for gid in io.archive_games()["RiotPlatformGameId"]:
        for f in io.load_v5(gid)["frames"][:-1]:
            t = f["timestamp"]
            if abs(t - 60000 * round(t / 60000)) > 2000:
                continue
            pf = f["participantFrames"]
            gd = sum(pf[str(i)]["totalGold"] for i in range(1, 6)) - sum(pf[str(i)]["totalGold"] for i in range(6, 11))
            rows.append((round(t / 60000), gd))
    return pd.DataFrame(rows, columns=["minute", "gd"]).groupby("minute")["gd"].std()


def boot(df, stat):
    """Point estimate, 95% series-block bootstrap interval, and two-sided bootstrap p-value vs 0."""
    pos = {k: np.asarray(v) for k, v in df.reset_index(drop=True).groupby("MatchId").indices.items()}
    keys = list(pos)
    base = df.reset_index(drop=True)
    est = stat(df)
    draws = np.array([stat(base.iloc[np.concatenate([pos[k] for k in rng.choice(keys, len(keys))])])
                      for _ in range(B)])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())) if len(draws) else np.nan
    return est, lo, hi, p


def bh(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.any():
        ps = p[ok]
        order = np.argsort(ps)
        ranked = ps[order] * len(ps) / (np.arange(len(ps)) + 1)
        qq = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(ps); out[order] = np.minimum(qq, 1)
        q[ok] = out
    return q


def md(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = [" | ".join(cols), " | ".join("---" for _ in cols)]
    rows += [" | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                        for v in r) for r in d.itertuples(index=False)]
    return "\n".join(rows)


def fmt(e, lo, hi, pct=True):
    if not np.isfinite(e):
        return "n/e"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
    return f"{f(e)} [{f(lo)}, {f(hi)}]"


def prepare():
    df = pd.read_parquet(io.ROOT / "fits" / "r_layer_setups.parquet")
    games = io.core_games()[["GameId", "MatchId"]]
    df = df.merge(games, on="GameId", how="left")
    sd = gold_sd_by_minute()
    pre = df[df["offset_s"] == -60][["RiotPlatformGameId", "objective", "side", "gold_diff", "tau3", "opp_tau3"]]
    pre = pre.rename(columns={"gold_diff": "gd_pre", "tau3": "tau3_pre", "opp_tau3": "opp_tau3_pre"})
    df = df.merge(pre, on=["RiotPlatformGameId", "objective", "side"], how="left")
    df["a"] = df["gd_pre"] / (((df["spawn_s"] - 60) / 60).round().astype(int)).map(sd)
    df["state"] = pd.cut(df["a"], [-np.inf, -0.5, 0.5, np.inf], labels=["trailing", "even", "leading"]).astype(str)
    for R in RADII:
        df[f"k_{R}"] = sum(((df[f"dpit_{r}"] <= R) & df[f"alive_{r}"]).astype(int) for r in io.ROLE_SHORT)
    opp = df[["RiotPlatformGameId", "objective", "offset_s", "side"] + [f"k_{R}" for R in RADII]].copy()
    opp["side"] = 300 - opp["side"]
    df = df.merge(opp.rename(columns={f"k_{R}": f"opp_k_{R}" for R in RADII}),
                  on=["RiotPlatformGameId", "objective", "offset_s", "side"], how="left")
    df["at_risk"] = (df["offset_s"] <= 0) | df["obj_up"]
    return df, sd


def main():
    out = io.RESULTS / "r_layer"
    out.mkdir(parents=True, exist_ok=True)
    df, sd = prepare()
    df.to_parquet(out / "r_layer_with_state.parquet", index=False)
    rs = df[df["at_risk"]]
    lines = ["# R-layer descriptive results (exact Riot frames; fixed-time first spawns)", "",
             f"Team-setups: {df.groupby(['RiotPlatformGameId', 'objective', 'side']).ngroups}; "
             f"series: {df['MatchId'].nunique()}; bootstrap: {B} series-block resamples; post-spawn rows use the "
             f"risk set (objective still alive); cells with n < {MIN_N} are 'n/e'. Primary pit radius {PRIMARY_R}.", ""]
    risk_n = rs[rs["offset_s"] == 60].groupby("objective").size().rename("s+60 risk-set team-setups")
    lines += ["Risk-set size at s+60 (of 328/322 team-setups per objective):", "", md(risk_n.to_frame()), ""]

    # 1. vocabulary (5 alive, risk set)
    full = rs[rs["alive"] == 5]
    voc = (full.groupby(["objective", "offset_s"])
               .agg(n=("partition", "size"), distinct_partitions=("partition", "nunique"),
                    distinct_shapes=("shape", "nunique")).reset_index())
    n_min = int(voc.loc[voc["n"] >= MIN_N, "n"].min())
    voc["rarefied_distinct_at_n_min"] = [
        np.mean([full[(full.objective == o) & (full.offset_s == t)]["partition"].sample(n_min, random_state=s).nunique()
                 for s in range(200)]) if n >= n_min else np.nan
        for o, t, n in zip(voc["objective"], voc["offset_s"], voc["n"])]
    voc.to_csv(out / "vocabulary.csv", index=False)
    (full.groupby(["objective", "offset_s"])["shape"].value_counts(normalize=True).rename("share").reset_index()
         .to_csv(out / "shape_shares.csv", index=False))
    (full.groupby(["objective", "offset_s"])["partition"].value_counts(normalize=True)
         .groupby(level=[0, 1]).head(5).rename("share").reset_index().to_csv(out / "top_partitions.csv", index=False))
    four1 = full[full["shape"] == "4+1"].copy()
    four1["single"] = four1["partition"].str.split("|").str[1]
    single_rows = []
    for r in io.ROLE_SHORT:
        e, lo, hi, _ = boot(four1, lambda d, r=r: (d["single"] == r).mean())
        single_rows.append({"role": r, "share": fmt(e, lo, hi), "uniform_null": "20.0%"})
    single = pd.DataFrame(single_rows)
    single.to_csv(out / "four_plus_one_singleton_role.csv", index=False)
    lines += ["## 1. Formation vocabulary (5 alive, risk set)", "", md(voc, index=False), "",
              f"Singleton role in 4+1 formations (n = {len(four1)} team-snapshots; series bootstrap):", "",
              md(single, index=False), ""]

    # 2 + 3. allocation and commitment by state, per radius, with paired trailing - leading contrasts
    stats = {
        "P(k=0)": lambda d, R: (d[f"k_{R}"] == 0).mean(),
        "P(k in {0,5})": lambda d, R: d[f"k_{R}"].isin([0, 5]).mean(),
        "mean k": lambda d, R: d[f"k_{R}"].mean(),
        "var k": lambda d, R: d[f"k_{R}"].var(),
        "P(self >= 3)": lambda d, R: (d[f"k_{R}"] >= 3).mean(),
        "P(both >= 3)": lambda d, R: ((d[f"k_{R}"] >= 3) & (d[f"opp_k_{R}"] >= 3)).mean(),
    }
    level_rows, contrast_rows = [], []
    for (obj, off), g in rs[rs["offset_s"] >= 0].groupby(["objective", "offset_s"]):
        for R in RADII:
            for name, fn in stats.items():
                for st in ("trailing", "even", "leading"):
                    h = g[g["state"] == st]
                    e, lo, hi, _ = boot(h, lambda d: fn(d, R)) if len(h) >= MIN_N else (np.nan,) * 4
                    level_rows.append({"objective": obj, "offset_s": off, "R": R, "stat": name, "state": st,
                                       "n": len(h), "est": e, "lo": lo, "hi": hi})
                tl = g[g["state"].isin(["trailing", "leading"])]
                ok = min((tl["state"] == "trailing").sum(), (tl["state"] == "leading").sum()) >= MIN_N
                e, lo, hi, p = boot(tl, lambda d: fn(d[d["state"] == "trailing"], R) - fn(d[d["state"] == "leading"], R)) \
                    if ok else (np.nan,) * 4
                contrast_rows.append({"objective": obj, "offset_s": off, "R": R, "stat": name,
                                      "trailing_minus_leading": e, "lo": lo, "hi": hi, "p_boot": p})
    lev = pd.DataFrame(level_rows)
    lev.to_csv(out / "allocation_by_state.csv", index=False)
    con = pd.DataFrame(contrast_rows)
    con["q_bh"] = bh(con["p_boot"])
    sign = (con.groupby(["objective", "offset_s", "stat"])["trailing_minus_leading"]
               .agg(lambda v: bool(v.notna().all() and (np.sign(v).nunique() == 1))).rename("sign_agrees_all_R"))
    con = con.merge(sign.reset_index(), on=["objective", "offset_s", "stat"])
    con["headline"] = con["sign_agrees_all_R"] & (con["q_bh"] < 0.05)
    con.to_csv(out / "contrasts_trailing_minus_leading.csv", index=False)
    lines += ["## 2-3. Allocation and commitment by gold state (primary R; series bootstrap 95% CI)", ""]
    for obj in OBJ_ORDER:
        for off in (0, 60):
            sub = lev[(lev.objective == obj) & (lev.offset_s == off) & (lev.R == PRIMARY_R)]
            if sub.empty:
                continue
            piv = sub.assign(v=[fmt(e, lo, hi, pct=not s.startswith(("mean", "var"))) for e, lo, hi, s in
                                zip(sub.est, sub.lo, sub.hi, sub.stat)]).pivot(index="stat", columns="state", values="v")
            ns = sub.groupby("state")["n"].first().to_dict()
            lines += [f"**{obj}, s{'+60' if off else ''}** (n: {ns})", "", md(piv[["trailing", "even", "leading"]]), ""]
    head = con[(con.R == PRIMARY_R)][["objective", "offset_s", "stat", "trailing_minus_leading", "lo", "hi",
                                       "q_bh", "sign_agrees_all_R", "headline"]]
    lines += ["### Trailing minus leading (paired within resamples; BH-FDR across all contrasts and radii)", "",
              md(head.dropna(subset=["trailing_minus_leading"]), index=False), ""]

    # 4. priority at s-60 vs outcome within the horizon (one row per spawn: blue-side perspective)
    pr = df[(df["offset_s"] == -60) & (df["side"] == 100)].copy()
    both_inf = np.isinf(pr["tau3_pre"]) & np.isinf(pr["opp_tau3_pre"])
    pr["priority"] = np.where(both_inf, 0, np.sign(pr["opp_tau3_pre"] - pr["tau3_pre"]))
    lat = pr["kill_s"] - pr["spawn_s"]
    in_h = pr["kill_s"].notna() & (lat <= C_HORIZON)
    pr["outcome"] = np.where(in_h, np.where(pr["taken_by_team"].eq(True), "self", "opp"), "neither")
    pri_rows = []
    for (obj, p), h in pr.groupby(["objective", "priority"]):
        for o in ("self", "opp", "neither"):
            e, lo, hi, _ = boot(h, lambda d, o=o: (d["outcome"] == o).mean()) if len(h) >= 15 else (np.nan,) * 4
            pri_rows.append({"objective": obj, "priority(blue)": int(p), "n_spawns": len(h), "outcome_within_180s": o,
                             "share": fmt(e, lo, hi)})
    pri = pd.DataFrame(pri_rows)
    pri.to_csv(out / "priority_vs_outcome.csv", index=False)
    lines += [f"## 4. Arrival priority at s-60 (3rd arrival, blue perspective) vs outcome within {C_HORIZON} s", "",
              "priority +1: blue can gather 3 at the pit sooner; grubs: owner = team with >= 2 of 3 grubs.", "",
              md(pri, index=False), ""]

    # 5. estimator audit at s+60, risk set, primary R: commit >= 3
    s60 = rs[(rs["offset_s"] == 60) & rs["state"].isin(["trailing", "leading"])].copy()
    s60["commit"] = (s60[f"k_{PRIMARY_R}"] >= 3).astype(float)

    def pooled(d, st):
        return d.loc[d["state"] == st, "commit"].mean()

    def team_mean(d, st):
        return d[d["state"] == st].groupby("team")["commit"].mean().mean()

    def within_team(d):
        m = d.groupby(["team", "state"])["commit"].mean().unstack()
        m = m.dropna(subset=["trailing", "leading"])
        return (m["trailing"] - m["leading"]).mean() if len(m) else np.nan

    audit = []
    for name, f in (("TAGNet-style unweighted team mean", team_mean), ("pooled (exposure-weighted)", pooled)):
        for st in ("trailing", "leading"):
            e, lo, hi, _ = boot(s60, lambda d, f=f, st=st: f(d, st))
            audit.append({"estimator": name, "quantity": st, "value": fmt(e, lo, hi)})
        e, lo, hi, p = boot(s60, lambda d, f=f: f(d, "trailing") - f(d, "leading"))
        audit.append({"estimator": name, "quantity": "trailing - leading", "value": fmt(e, lo, hi) + f" (p={p:.3f})"})
    e, lo, hi, p = boot(s60, within_team)
    audit.append({"estimator": "within-team (teams seen in both states)", "quantity": "trailing - leading",
                  "value": fmt(e, lo, hi) + f" (p={p:.3f})"})
    audit = pd.DataFrame(audit)
    audit.to_csv(out / "estimator_audit.csv", index=False)
    lines += [f"## 5. Estimator audit: P(>= 3 at the pit at s+60), risk set, R = {PRIMARY_R}, all objectives", "",
              md(audit, index=False), ""]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
