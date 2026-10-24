"""PILOT: team formations as set partitions of the 5 roles, from Riot V5 60-s frames (Worlds 2025).

Deliberately crude (Euclidean distance, 60-s frames, approximate dead-player exclusion) -- it only
shows what the formation vocabulary and the 'trailing vs leading' table look like in real data.
The paper version uses geodesic distance on the navigation graph and 2-fps tracker snapshots.

Outputs: results/pilot_formations/*.csv
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data" / "raw" / "v5"
OUT = ROOT / "results" / "pilot_formations"
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
R_GROUP = 2000          # map units: teammates within r are linked (single linkage)
MAP_DIAG = 14870        # river ~ anti-diagonal x + y = MAP_DIAG (blue base at bottom-left)
GOLD_LEAD = 2000
DEAD_WINDOW_MS = 60_000  # crude: drop a team-frame if a member died within the previous 60 s
N_BOOT = 2000
rng = np.random.default_rng(0)


def load(kind, gid):
    with gzip.open(V5 / kind / f"{gid}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def partition(points, r):
    """Connected components of the r-proximity graph -> set partition of role indices."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(points[i] - points[j])) <= r:
                parent[find(i)] = find(j)
    blocks = {}
    for i in range(n):
        blocks.setdefault(find(i), []).append(i)
    return sorted((tuple(sorted(b)) for b in blocks.values()), key=lambda b: (-len(b), b))


def shape(blocks):
    return "+".join(str(len(b)) for b in blocks)


def label(blocks):
    return " | ".join("{" + ",".join(ROLES[i][0] if ROLES[i] != "Jungle" else "J" for i in b) + "}" for b in blocks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(ROOT / "data" / "manifests" / "core_worlds2025.csv")
    sp = pd.read_csv(ROOT / "data" / "raw" / "leaguepedia" / "lck_intl__ScoreboardPlayers.csv")

    rows, role_checks = [], []
    for _, g in core.iterrows():
        gid = g["RiotPlatformGameId"]
        tl, pg = load("timeline", gid), load("postgame", gid)
        parts = sorted(pg["participants"], key=lambda p: p["participantId"])
        teams = {100: g["Blue"], 200: g["Red"]}

        # verify 'participant order = role order' against Leaguepedia roles
        lp = sp[sp["GameId"] == g["GameId"]]
        for p in parts:
            name = (p.get("riotIdGameName") or p.get("summonerName") or "").split(" ", 1)[-1].lower()
            hit = lp[lp["Link"].astype(str).str.lower().str.split(" ").str[0] == name.split(" ")[0]]
            if len(hit):
                role_checks.append(ROLES[(p["participantId"] - 1) % 5] == hit.iloc[0]["Role"])

        deaths = [(e["timestamp"], e["victimId"]) for f in tl["frames"] for e in f["events"]
                  if e["type"] == "CHAMPION_KILL"]
        for f in tl["frames"]:
            t = f["timestamp"]
            minute = t / 60000
            if minute < 3:
                continue
            pf = f["participantFrames"]
            gold = {100: sum(pf[str(i)]["totalGold"] for i in range(1, 6)),
                    200: sum(pf[str(i)]["totalGold"] for i in range(6, 11))}
            for side, ids in ((100, range(1, 6)), (200, range(6, 11))):
                if any(v in ids and 0 <= t - ts <= DEAD_WINDOW_MS for ts, v in deaths):
                    continue
                pts = np.array([[pf[str(i)]["position"]["x"], pf[str(i)]["position"]["y"]] for i in ids], float)
                blocks = partition(pts, R_GROUP)
                depth = pts.sum(axis=1) if side == 100 else 2 * MAP_DIAG - pts.sum(axis=1)
                gd = gold[side] - gold[300 - side]
                rows.append({
                    "GameId": g["GameId"], "MatchId": g["MatchId"], "team": teams[side], "side": side,
                    "minute": round(minute), "gold_diff": gd,
                    "gold_state": "leading" if gd >= GOLD_LEAD else "trailing" if gd <= -GOLD_LEAD else "even",
                    "phase": "03-13" if minute < 14 else "14-24" if minute < 25 else "25+",
                    "shape": shape(blocks), "partition": label(blocks),
                    "n_forward": int((depth > MAP_DIAG).sum()),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "team_frames.csv", index=False)

    # 1) formation vocabulary: shapes and how many of the 52 set partitions occur
    voc = df.groupby("shape").size().rename("team_frames").reset_index()
    voc["share"] = (voc["team_frames"] / voc["team_frames"].sum()).round(3)
    voc.sort_values("team_frames", ascending=False).to_csv(OUT / "shape_vocabulary.csv", index=False)
    top_parts = df["partition"].value_counts().head(15).rename_axis("partition").reset_index(name="team_frames")
    top_parts.to_csv(OUT / "top_partitions.csv", index=False)

    # 2) shape distribution by gold state and phase
    tab = (df.groupby(["phase", "gold_state", "shape"]).size()
             .groupby(level=[0, 1]).transform(lambda s: s / s.sum()).round(3).unstack("shape").fillna(0))
    tab.to_csv(OUT / "shape_by_gold_state_phase.csv")

    # 3) TAGNet analogue: P(>=2 players past the river) leading vs trailing,
    #    pooled with game-cluster bootstrap CI vs TAGNet-style unweighted team mean
    df["fwd2"] = (df["n_forward"] >= 2).astype(int)
    out = []
    for phase in ["03-13", "14-24", "25+"]:
        for st in ["leading", "even", "trailing"]:
            d = df[(df.phase == phase) & (df.gold_state == st)]
            if d.empty:
                continue
            games = d["GameId"].unique()
            by_game = d.groupby("GameId")["fwd2"].agg(["sum", "count"])
            boots = []
            for _ in range(N_BOOT):
                s = by_game.loc[rng.choice(games, len(games))]
                boots.append(s["sum"].sum() / s["count"].sum())
            out.append({
                "phase": phase, "gold_state": st, "team_frames": len(d), "games": len(games),
                "pooled_rate": round(d["fwd2"].mean(), 3),
                "ci95_lo": round(np.percentile(boots, 2.5), 3), "ci95_hi": round(np.percentile(boots, 97.5), 3),
                "tagnet_style_team_mean": round(d.groupby("team")["fwd2"].mean().mean(), 3),
                "teams": d["team"].nunique(),
            })
    pd.DataFrame(out).to_csv(OUT / "forward_presence_by_gold_state.csv", index=False)

    print(f"team-frames: {len(df)} from {df['GameId'].nunique()} games; "
          f"role-order check: {np.mean(role_checks):.3f} agreement over {len(role_checks)} players")
    print(f"distinct set partitions observed: {df['partition'].nunique()} of 52; shapes: {df['shape'].nunique()} of 7")
    print(voc.sort_values("team_frames", ascending=False).to_string(index=False))
    print(top_parts.head(8).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
