"""V-layer descriptive analysis (2-fps minimap tracks; every spawn of the core games; s-120 .. s+60 every 15 s).

Time-resolved version of analysis_r with the same statistics and rules:
  1. commitment curves: P(k = 0), mean k and P(k >= 3) at every offset, by objective and gold state
  2. trailing - leading contrasts at every offset (paired inside each series-block resample); BH-FDR over all
     (objective, offset, stat, R) cells; headline = sign agrees at all three pit radii and q < 0.05
  3. arrival race: the first offset at which each team has >= 3 alive players within R of the pit, and who
     takes the objective within C_HORIZON of the spawn (one row per spawn)
Objective groups: the five fixed-time first spawns, plus DRAGON_2+ (respawned dragons), BARON_2+ and ELDER.
Gold state: GD at s-120 (first grid point) / archive SD of GD at that minute; trailing < -0.5 SD < even < 0.5 SD.
Post-spawn rows use the risk set (objective still alive). Sensitivity: the same contrasts without
team-snapshots whose alive positions are mostly held or interpolated (q_hold + q_interp > 0.5).

Usage:  python -m lolsetup.analysis_v  -> results/v_layer/*.csv + summary.md
"""
import numpy as np
import pandas as pd

from . import io
from .analysis_r import B, MIN_N, C_HORIZON, RADII, PRIMARY_R, bh, boot, fmt, gold_sd_by_minute, md

GROUPS = ["DRAGON", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR", "DRAGON_2+", "BARON_2+", "ELDER"]
KEY = ["RiotPlatformGameId", "objective", "idx", "side"]


def group_of(objective, idx):
    if objective in ("DRAGON", "BARON_NASHOR") and idx > 1:
        return f"{'DRAGON' if objective == 'DRAGON' else 'BARON'}_2+"
    return objective


def prepare():
    df = pd.read_parquet(io.ROOT / "fits" / "v_layer_setups.parquet")
    df = df.merge(io.core_games()[["GameId", "MatchId"]], on="GameId", how="left")
    df["group"] = [group_of(o, i) for o, i in zip(df["objective"], df["idx"])]
    sd = gold_sd_by_minute()
    first = df.sort_values("offset_s").groupby(KEY).first()[["gold_diff", "game_s"]]
    first = first.rename(columns={"gold_diff": "gd_pre", "game_s": "t_pre"}).reset_index()
    df = df.merge(first, on=KEY, how="left")
    df["a"] = df["gd_pre"] / (df["t_pre"] / 60).round().astype(int).clip(upper=sd.index.max()).map(sd)
    df["state"] = pd.cut(df["a"], [-np.inf, -0.5, 0.5, np.inf], labels=["trailing", "even", "leading"]).astype(str)
    for R in RADII:
        df[f"k_{R}"] = sum(((df[f"dpit_{r}"] <= R) & df[f"alive_{r}"]).astype(int) for r in io.ROLE_SHORT)
    opp = df[["RiotPlatformGameId", "objective", "idx", "offset_s", "side"] + [f"k_{R}" for R in RADII]].copy()
    opp["side"] = 300 - opp["side"]
    df = df.merge(opp.rename(columns={f"k_{R}": f"opp_k_{R}" for R in RADII}),
                  on=["RiotPlatformGameId", "objective", "idx", "offset_s", "side"], how="left")
    df["at_risk"] = (df["offset_s"] <= 0) | df["obj_up"]
    return df


STATS = {
    "P(k=0)": lambda d, R: (d[f"k_{R}"] == 0).mean(),
    "mean k": lambda d, R: d[f"k_{R}"].mean(),
    "P(self >= 3)": lambda d, R: (d[f"k_{R}"] >= 3).mean(),
}


def raw_curves(rs, groups=None):
    """Levels and trailing-minus-leading contrasts with bootstrap intervals for the given objective groups
    (no multiplicity step: BH-FDR runs over all cells in finalize)."""
    level_rows, contrast_rows = [], []
    for (grp, off), g in rs.groupby(["group", "offset_s"]):
        if groups is not None and grp not in groups:
            continue
        for R in RADII:
            for name, fn in STATS.items():
                for st in ("trailing", "even", "leading"):
                    h = g[g["state"] == st]
                    e, lo, hi, _ = boot(h, lambda d: fn(d, R)) if len(h) >= MIN_N else (np.nan,) * 4
                    level_rows.append({"group": grp, "offset_s": off, "R": R, "stat": name, "state": st,
                                       "n": len(h), "est": e, "lo": lo, "hi": hi})
                tl = g[g["state"].isin(["trailing", "leading"])]
                ok = min((tl["state"] == "trailing").sum(), (tl["state"] == "leading").sum()) >= MIN_N
                e, lo, hi, p = boot(tl, lambda d: fn(d[d["state"] == "trailing"], R) - fn(d[d["state"] == "leading"], R)) \
                    if ok else (np.nan,) * 4
                contrast_rows.append({"group": grp, "offset_s": off, "R": R, "stat": name,
                                      "trailing_minus_leading": e, "lo": lo, "hi": hi, "p_boot": p})
    return pd.DataFrame(level_rows), pd.DataFrame(contrast_rows)


def finalize(con):
    con = con.copy()
    con["q_bh"] = bh(con["p_boot"])
    sign = (con.groupby(["group", "offset_s", "stat"])["trailing_minus_leading"]
               .agg(lambda v: bool(v.notna().all() and (np.sign(v).nunique() == 1))).rename("sign_agrees_all_R"))
    con = con.merge(sign.reset_index(), on=["group", "offset_s", "stat"])
    con["headline"] = con["sign_agrees_all_R"] & (con["q_bh"] < 0.05)
    return con


def curves_and_contrasts(rs):
    lev, con = raw_curves(rs)
    return lev, finalize(con)


def arrival_race(df, R=PRIMARY_R):
    """First offset with >= 3 at the pit per team (inf if never on the grid) and the outcome within C_HORIZON."""
    rows = []
    for key, g in df.groupby(["RiotPlatformGameId", "objective", "idx"]):
        g = g[g["at_risk"]]
        first = {}
        for side in (100, 200):
            h = g[(g["side"] == side) & (g[f"k_{R}"] >= 3)]
            first[side] = h["offset_s"].min() if len(h) else np.inf
        r0 = g.iloc[0] if len(g) else None
        if r0 is None:
            continue
        lat = r0["kill_s"] - r0["spawn_s"]
        taken = r0["kill_s"] == r0["kill_s"] and lat <= C_HORIZON
        blue_took = bool(g.loc[g["side"] == 100, "taken_by_team"].eq(True).any())
        rows.append({"RiotPlatformGameId": key[0], "group": r0["group"], "MatchId": r0["MatchId"],
                     "first3_blue": first[100], "first3_red": first[200],
                     "race": "blue" if first[100] < first[200] else "red" if first[200] < first[100] else "tie",
                     "outcome": ("blue" if blue_took else "red") if taken else "neither"})
    return pd.DataFrame(rows)


def cmd_part(group, clean):
    """One (objective group, full|clean) slice of the bootstrap; merged by main(merge=True)."""
    import pickle
    from . import analysis_r
    analysis_r.rng = np.random.default_rng(20261022 + 97 * GROUPS.index(group) + 13 * int(clean))
    df = prepare()
    rs = df[df["at_risk"]]
    if clean:
        rs = rs[(rs["q_hold"].fillna(0) + rs.get("q_interp", 0)) <= 0.5]
    lev, con = raw_curves(rs, [group])
    d = io.ROOT / "fits" / "vparts"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{group}_{int(clean)}.pkl", "wb") as f:
        pickle.dump((lev, con), f)
    print(group, clean, len(lev), len(con))


def main(merge=False):
    out = io.RESULTS / "v_layer"
    out.mkdir(parents=True, exist_ok=True)
    df = prepare()
    df.to_parquet(out / "v_layer_with_state.parquet", index=False)
    rs = df[df["at_risk"]]
    n_set = df.groupby(KEY).ngroups
    lines = ["# V-layer descriptive results (2-fps minimap tracks; every spawn; s-120..s+60 every 15 s)", "",
             f"Team-setups: {n_set}; games: {df['RiotPlatformGameId'].nunique()}; series: {df['MatchId'].nunique()}; "
             f"bootstrap: {B} series-block resamples; risk set after the spawn; cells with n < {MIN_N} are 'n/e'. "
             f"Primary pit radius {PRIMARY_R}.", "",
             "Team-setups per objective group:", "", md(df.groupby("group").apply(lambda d: d.groupby(KEY).ngroups)
                                                       .rename("team-setups").to_frame()), "",
             "Position sources (share of alive positions): detected / held (covered) / pinned (V5 minute):", "",
             md(df.groupby("group")[["q_det", "q_hold", "q_pin"]].mean()), ""]

    if merge:
        import pickle
        parts = {c: [pickle.load(open(io.ROOT / "fits" / "vparts" / f"{g}_{c}.pkl", "rb")) for g in GROUPS] for c in (0, 1)}
        lev = pd.concat([p[0] for p in parts[0]], ignore_index=True)
        con = finalize(pd.concat([p[1] for p in parts[0]], ignore_index=True))
    else:
        lev, con = curves_and_contrasts(rs)
    lev.to_csv(out / "curves_by_state.csv", index=False)
    con.to_csv(out / "contrasts_trailing_minus_leading.csv", index=False)
    head = con[con["headline"] & (con["R"] == PRIMARY_R)]
    lines += ["## 1-2. Commitment curves and trailing - leading contrasts", "",
              f"Headline cells (sign agrees at R in {RADII}, BH-FDR q < 0.05; shown at R = {PRIMARY_R}):", "",
              md(head[["group", "offset_s", "stat", "trailing_minus_leading", "lo", "hi", "q_bh"]], index=False)
              if len(head) else "(none)", ""]

    clean = rs[(rs["q_hold"].fillna(0) + rs.get("q_interp", 0)) <= 0.5]
    con_c = finalize(pd.concat([p[1] for p in parts[1]], ignore_index=True)) if merge else curves_and_contrasts(clean)[1]
    con_c.to_csv(out / "contrasts_trailing_minus_leading_clean.csv", index=False)
    both = con[["group", "offset_s", "R", "stat", "headline", "trailing_minus_leading"]].merge(
        con_c[["group", "offset_s", "R", "stat", "headline", "trailing_minus_leading"]],
        on=["group", "offset_s", "R", "stat"], suffixes=("", "_clean"))
    kept = both[both["headline"]]
    lines += [f"Sensitivity without mostly-held/interpolated snapshots (q_hold + q_interp > 0.5 dropped; {len(rs) - len(clean)} of {len(rs)} "
              f"rows): {int(kept['headline_clean'].sum())} of {len(kept)} headline cells remain headlines; "
              f"sign kept in {int((np.sign(kept['trailing_minus_leading']) == np.sign(kept['trailing_minus_leading_clean'])).sum())}.", ""]

    race = arrival_race(df)
    race.to_csv(out / "arrival_race.csv", index=False)
    tab = (race[race["race"] != "tie"].assign(winner_took=lambda d: d["race"] == d["outcome"])
               .groupby("group").agg(n=("race", "size"), race_winner_took=("winner_took", "mean"),
                                     neither=("outcome", lambda v: (v == "neither").mean())))
    lines += [f"## 3. Arrival race: first team with >= 3 at the pit (R = {PRIMARY_R}) vs taker within {C_HORIZON} s", "",
              md(tab), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "part", "merge"])
    ap.add_argument("--group", default=None)
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    if a.cmd == "part":
        cmd_part(a.group, a.clean)
    else:
        main(merge=(a.cmd == "merge"))
