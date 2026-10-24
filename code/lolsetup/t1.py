"""T1 / Faker appendix analyses (plan section 8) as far as the data allow. Descriptive, series-block bootstrap
intervals, compared with the same measure for the other teams / other starting mid laners.

    python -m lolsetup.t1            -> results/t1/*.csv, results/t1/summary.md

A1  pattern fingerprint: results/fingerprints (fingerprints.py); copied nearest-neighbour table when available.
A2  QRE parameters of T1 among the league: results/fingerprints/theta.csv and the bootstrap in results/robustness.
A3  Faker's position in the setups: pit presence, arrival rank among the five, alone-vs-main-block, MST degree and
    betweenness at the spawn snapshot, against every other starting mid laner (Oracle's Elixir player names).
A4  mid-lane effect u^M: not estimated in this version of the model (no role layer); A3 is the descriptive substitute.
A5  behaviour when trailing / leading: commitment share, polarisation P(k in {0, 5}) and the matched commitment
    contrast (trailing minus leading), T1 against the other teams.
A6  fight-function shift: T1's average residual (outcome - fitted Tullock probability) in tracked fights.
A7  decision ledger of the Worlds 2025 final (T1 vs KT) spawn by spawn: state, actions, fight probability,
    break-even probability, realised objective owner.
A8  MSI 2025 vs Worlds 2025 for T1 (same rules): commitment share by event.
A9  rule-different control (Worlds 2023/24, T1): the control videos are not tracked in this version -> not available.
A10 narrative-to-measure map (text table).
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import expit

from . import io, qre_data, run_qre, run_robust
from .context import KEY

OUT = io.RESULTS / "t1"
TEAM = "T1"


def oe_mid():
    """gameid, side -> mid laner name (Oracle's Elixir)."""
    frames = []
    for y in (2025,):
        d = pd.read_csv(io.DATA / "raw" / "oracles_elixir" / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False,
                        usecols=["gameid", "position", "playername", "teamname", "side"])
        frames.append(d[d["position"] == "mid"])
    d = pd.concat(frames)
    d["side"] = d["side"].map({"Blue": 100, "Red": 200})
    return d.rename(columns={"gameid": "RiotPlatformGameId", "playername": "mid_player"})[["RiotPlatformGameId", "side", "mid_player"]]


def mid_measures(v, offsets=(-60, -45, -30, -15, 0)):
    """Per team-setup: mean over the pre-spawn snapshots of mid-laner position measures."""
    d = v[v["offset_s"].isin(offsets)]
    rows = []
    for key, g in d.groupby(KEY):
        vals = {"in_pit": [], "rank": [], "alone": [], "main": [], "deg": [], "btw": [], "dpit": []}
        for r in g.itertuples():
            if not getattr(r, "alive_M"):
                continue
            alive = [c for c in io.ROLE_SHORT if getattr(r, f"alive_{c}")]
            dp = np.array([getattr(r, f"dpit_{c}") for c in alive])
            vals["in_pit"].append(float(getattr(r, "dpit_M") <= 1500))
            vals["dpit"].append(float(getattr(r, "dpit_M")))
            vals["rank"].append(float((dp < getattr(r, "dpit_M")).sum() + 1))
            blocks = r.partition.split("|")
            vals["alone"].append(float("M" in blocks))
            big = max(blocks, key=len)
            vals["main"].append(float("M" in big and len(big) >= 2))
            if len(alive) >= 3:
                xy = np.array([[getattr(r, f"x_{c}"), getattr(r, f"y_{c}")] for c in alive])
                D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
                T = minimum_spanning_tree(D).toarray()
                G = nx.from_numpy_array(np.maximum(T, T.T))
                i = alive.index("M")
                vals["deg"].append(float(G.degree(i)))
                vals["btw"].append(float(nx.betweenness_centrality(G)[i]))
        if vals["in_pit"]:
            rows.append({**dict(zip(KEY, key)), **{k: float(np.mean(x)) if x else np.nan for k, x in vals.items()}})
    return pd.DataFrame(rows)


def boot_diff(d, flag, cols, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    series = d["MatchId"].to_numpy()
    uniq = np.unique(series)
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    f = d[flag].to_numpy(bool)

    def diff(ii):
        x = d.iloc[ii]
        m = f[ii]
        return np.array([x.loc[m, c].mean() - x.loc[~m, c].mean() for c in cols])
    pt = diff(np.arange(len(d)))
    reps = np.array([diff(np.concatenate([idx_by[s] for s in rng.choice(uniq, len(uniq))])) for _ in range(B)])
    return pd.DataFrame({"measure": cols, "flagged_mean": [d.loc[f, c].mean() for c in cols], "others_mean": [d.loc[~f, c].mean() for c in cols],
                         "diff": pt, "lo": np.nanquantile(reps, 0.025, 0), "hi": np.nanquantile(reps, 0.975, 0),
                         "n_flagged": int(f.sum()), "n_others": int((~f).sum())})


def main(boot=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(io.ROOT / "fits" / "v_layer_setups.parquet")
    games = io.core_games()[["GameId", "MatchId"]]
    cov = pd.read_parquet(io.RESULTS / "context" / "context_covariates.parquet") if (io.RESULTS / "context" / "context_covariates.parquet").exists() else None
    md = ["# T1 / Faker appendix (descriptive; series-block bootstrap 95% intervals)", ""]
    # ---- A3
    mm = mid_measures(v)
    first = v.drop_duplicates(KEY)[KEY + ["GameId", "team", "event"]].merge(games, on="GameId", how="left")
    mm = mm.merge(first, on=KEY).merge(oe_mid(), on=["RiotPlatformGameId", "side"], how="left")
    mm["MatchId"] = mm["MatchId"].astype(str)
    mm["faker"] = (mm["mid_player"] == "Faker")
    mm.to_csv(OUT / "mid_measures.csv", index=False)
    cols = ["in_pit", "rank", "dpit", "alone", "main", "deg", "btw"]
    a3 = boot_diff(mm.dropna(subset=["mid_player"]), "faker", cols, boot)
    a3.to_csv(OUT / "A3_faker_vs_other_mids.csv", index=False)
    mm["t1"] = mm["team"] == TEAM
    a3b = boot_diff(mm, "t1", cols, boot)
    a3b.to_csv(OUT / "A3b_t1_vs_other_teams_mid.csv", index=False)
    md += ["## A3 Faker vs other starting mid laners (spawn-anchored pre-spawn snapshots s-60..s)", "", a3.round(3).to_string(index=False), "",
           f"Faker setups: {int(mm['faker'].sum())}; all mid-lane setups: {len(mm)}.", ""]
    # ---- A5 / A8 / A7 from the QRE contest table
    P = run_qre.prepare("wpa")
    d = run_robust.team_contests(P)
    d.to_csv(OUT / "team_contests_with_names.csv", index=False)
    d["polar"] = d["k"].isin([0, 5]).astype(int)
    d["t1"] = d["team"] == TEAM
    rows = []
    for st in ("trailing", "even", "leading"):
        for who, m in (("T1", d["t1"]), ("others", ~d["t1"])):
            x = d[m & (d["state"] == st)]
            rows.append({"group": who, "state": st, "n": len(x), "commit_share": x["commit"].mean(), "polarisation": x["polar"].mean(),
                         "mean_k": x["k"].mean()})
    a5 = pd.DataFrame(rows)
    a5.to_csv(OUT / "A5_state_behaviour.csv", index=False)
    series = d["MatchId"].to_numpy()
    uniq = np.unique(series)
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    rng = np.random.default_rng(1)

    def gap(x, m):
        a, b = x[m & (x["state"] == "trailing")], x[m & (x["state"] == "leading")]
        return a["commit"].mean() - b["commit"].mean()
    pt = {w: gap(d, m) for w, m in (("T1", d["t1"]), ("others", ~d["t1"]))}
    pt["T1_minus_others"] = pt["T1"] - pt["others"]
    reps = []
    for _ in range(boot):
        x = d.iloc[np.concatenate([idx_by[s] for s in rng.choice(uniq, len(uniq))])]
        g1, g0 = gap(x, x["t1"]), gap(x, ~x["t1"])
        reps.append((g1, g0, g1 - g0))
    reps = np.array(reps)
    a5b = pd.DataFrame({"contrast": ["T1 trailing-minus-leading commit share", "others", "T1 minus others"],
                        "estimate": [pt["T1"], pt["others"], pt["T1_minus_others"]],
                        "lo": np.nanquantile(reps, 0.025, 0), "hi": np.nanquantile(reps, 0.975, 0)})
    a5b.to_csv(OUT / "A5_trailing_minus_leading_contrast.csv", index=False)
    md += ["## A5 behaviour by state", "", a5.round(3).to_string(index=False), "", a5b.round(3).to_string(index=False), ""]
    ev = P["info"][["RiotPlatformGameId"]].copy()
    d2 = d.copy()
    d2["event"] = P["info"]["RiotPlatformGameId"].map(first.drop_duplicates("RiotPlatformGameId").set_index("RiotPlatformGameId")["event"]).repeat(2).to_numpy()
    a8 = d2[d2["team"] == TEAM].groupby("event").agg(n=("commit", "size"), commit_share=("commit", "mean"), mean_k=("k", "mean")).reset_index()
    a8.to_csv(OUT / "A8_t1_by_event.csv", index=False)
    md += ["## A8 T1 by event (same rules)", "", a8.round(3).to_string(index=False), ""]
    # ---- A6 fight shift
    f = pd.read_parquet(io.ROOT / "fits" / "fights.parquet")
    law = run_qre.outcome_params("tracked")
    g = games.merge(io.core_games()[["GameId", "RiotPlatformGameId", "Blue", "Red"]], on="GameId")
    f = f.merge(g[["RiotPlatformGameId", "Blue", "Red", "MatchId"]], on="RiotPlatformGameId", how="left")
    f = f[f["n_blue_tracked"].gt(0) & f["n_red_tracked"].gt(0) & f["winner"].isin([100, 200])]
    lin = law.gamma_n * np.log(f["n_blue_tracked"] / f["n_red_tracked"]) + law.gamma_g * np.log(f["gold_pp_blue"] / f["gold_pp_red"])
    f["resid_blue"] = (f["winner"] == 100).astype(float) - expit(lin)
    t1 = np.where(f["Blue"] == TEAM, f["resid_blue"], np.where(f["Red"] == TEAM, -f["resid_blue"], np.nan))
    f["t1_resid"] = t1
    x = f.dropna(subset=["t1_resid"])
    ser = x["MatchId"].astype(str).to_numpy()
    uq = np.unique(ser)
    ib = {s: np.flatnonzero(ser == s) for s in uq}
    bs = [x["t1_resid"].to_numpy()[np.concatenate([ib[s] for s in rng.choice(uq, len(uq))])].mean() for _ in range(boot)]
    a6 = pd.DataFrame([{"n_fights_with_T1": len(x), "mean_residual": x["t1_resid"].mean(), "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975)}])
    a6.to_csv(OUT / "A6_fight_shift.csv", index=False)
    md += ["## A6 fight-function shift (T1 outcome minus fitted Tullock probability)", "", a6.round(4).to_string(index=False), ""]
    # ---- A7 ledger of the Worlds 2025 final
    fin = io.core_games()
    fin = fin[fin["GameId"].str.contains("Main Event_Finals")]["RiotPlatformGameId"].tolist()
    info = P["info"].reset_index(drop=True)
    rows = []
    for i, (s, (a, b)) in enumerate(zip(P["spec"], P["choices"])):
        r = info.iloc[i]
        if r["RiotPlatformGameId"] not in fin:
            continue
        ab, ar = run_qre.qre.actions(s["n_b"], s["kb"])[a], run_qre.qre.actions(s["n_r"], s["kr"])[b]
        v_ = s["vals"]
        p_hat = float(expit(P["oc"].gamma_n * np.log(s["n_b"] / s["n_r"]) + P["oc"].gamma_g * s["lng"]))
        p_star = float(-v_.fight_lose / (v_.obj_self + v_.fight_win - v_.obj_opp - v_.fight_lose))
        rows.append({"game": r["RiotPlatformGameId"], "objective": r["objective"], "idx": r["idx"], "minute": r["minute"], "W_blue": r["W_blue"],
                     "gold_diff_blue": r["gd"], "n_blue": s["n_b"], "n_red": s["n_r"], "blue_k": int(ab[0]), "blue_m": int(ab[1]),
                     "blue_tau": int(ab[2]), "red_k": int(ar[0]), "red_m": int(ar[1]), "red_tau": int(ar[2]),
                     "p_hat_blue": p_hat, "p_star_blue_break_even": p_star, "value_obj_blue": v_.obj_self, "value_obj_red": v_.obj_opp})
    a7 = pd.DataFrame(rows)
    spw = io.spawns()
    a7 = a7.merge(spw[["RiotPlatformGameId", "objective", "idx", "taken_by", "kill_s", "spawn_s"]].rename(columns={"game": "RiotPlatformGameId"}),
                  left_on=["game", "objective", "idx"], right_on=["RiotPlatformGameId", "objective", "idx"], how="left").drop(columns="RiotPlatformGameId")
    a7.to_csv(OUT / "A7_worlds2025_final_ledger.csv", index=False)
    md += ["## A7 Worlds 2025 final ledger", "", f"{len(a7)} objective spawns across the five games; see A7_worlds2025_final_ledger.csv.", ""]
    md += ["## A9 / A10", "", "A9 (Worlds 2023/24, different objective rules): the control videos were not processed by the tracker in this version; "
           "not available. A4 (role effect u^M) is not estimated; A3 gives the descriptive position measures instead. "
           "A1/A2: see results/fingerprints.", "",
           "| narrative | measure here | status |", "|---|---|---|",
           "| 'T1 does not panic when behind' | A5 trailing-minus-leading commitment contrast, T1 vs others | measurable |",
           "| 'Faker controls the map' | A3 position measures (arrival rank, block membership, MST centrality) | measurable |",
           "| 'T1 wins fights at equal numbers' | A6 residual of the fight function | measurable, descriptive |",
           "| champion-pool depth, draft mind games, communication | no champion / audio data in this study | not measurable |"]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
