"""PILOT: how teams set up in the minutes before neutral objectives (Worlds 2025, Riot V5 60-s frames).

Two views:
  (a) kill-anchored: every taken objective; the two frames before the kill (0-60 s and 60-120 s before)
  (b) spawn-anchored: first dragon (fixed spawn time), frames at spawn-120 s, -60 s and spawn
Crude on purpose (Euclidean distance, approximate dead-player handling, taken objectives only in (a)).
The paper version anchors every spawn (taken or not), uses geodesic distance and 2-fps tracker snapshots.

Outputs: results/pilot_objective_setups/*.csv
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_formations import partition, shape

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data" / "raw" / "v5"
OUT = ROOT / "results" / "pilot_objective_setups"
NEAR = 3000             # map units from the pit
GROUP_R = 2000
GOLD_LEAD = 2000
DEAD_WINDOW_MS = 45_000
FIRST_DRAGON_SPAWN_MS = 5 * 60_000  # to be confirmed against patch 25.20 rules


def load(kind, gid):
    with gzip.open(V5 / kind / f"{gid}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def objective_instances(events):
    """ELITE_MONSTER_KILL events, with the 3 void grubs of one spawn merged into one instance."""
    inst = []
    for e in sorted(events, key=lambda e: e["timestamp"]):
        kind = e["monsterType"] if e["monsterType"] != "DRAGON" else (
            "ELDER" if e.get("monsterSubType") == "ELDER_DRAGON" else "DRAGON")
        if kind == "HORDE" and inst and inst[-1]["kind"] == "HORDE" and e["timestamp"] - inst[-1]["t_last"] < 120_000:
            inst[-1]["t_last"] = e["timestamp"]
            inst[-1]["takers"].add(e["killerTeamId"])
            continue
        inst.append({"kind": kind, "t": e["timestamp"], "t_last": e["timestamp"], "takers": {e["killerTeamId"]},
                     "x": e["position"]["x"], "y": e["position"]["y"]})
    return inst


def team_view(frame, ids, pit, dead_ids):
    alive = [i for i in ids if i not in dead_ids]
    pts = np.array([[frame["participantFrames"][str(i)]["position"]["x"],
                     frame["participantFrames"][str(i)]["position"]["y"]] for i in alive], float)
    d = np.hypot(pts[:, 0] - pit[0], pts[:, 1] - pit[1]) if len(pts) else np.array([])
    return {"alive": len(alive), "near": int((d <= NEAR).sum()),
            "shape": shape(partition(pts, GROUP_R)) if len(pts) else ""}


def gold(frame, ids):
    return sum(frame["participantFrames"][str(i)]["totalGold"] for i in ids)


def dead_at(t, deaths):
    return {v for ts, v in deaths if 0 <= t - ts <= DEAD_WINDOW_MS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(ROOT / "data" / "manifests" / "core_worlds2025.csv")
    sides = {100: range(1, 6), 200: range(6, 11)}
    rows, spawn_rows, first_kill = [], [], {}

    for _, g in core.iterrows():
        tl = load("timeline", g["RiotPlatformGameId"])
        frames = tl["frames"]
        events = [e for f in frames for e in f["events"]]
        deaths = [(e["timestamp"], e["victimId"]) for e in events if e["type"] == "CHAMPION_KILL"]
        names = {100: g["Blue"], 200: g["Red"]}
        insts = objective_instances([e for e in events if e["type"] == "ELITE_MONSTER_KILL"])
        for kind in {i["kind"] for i in insts}:
            first_kill.setdefault(kind, []).append(min(i["t"] for i in insts if i["kind"] == kind) / 1000)

        # (a) kill-anchored setups
        for k, inst in enumerate(insts):
            for lag, label in ((0, "0-60s before"), (1, "60-120s before")):
                fi = int(inst["t"] // 60_000) - lag
                if fi < 1 or fi >= len(frames):
                    continue
                fr = frames[fi]
                dead = dead_at(fr["timestamp"], deaths)
                for side in (100, 200):
                    v = team_view(fr, sides[side], (inst["x"], inst["y"]), dead)
                    gd = gold(fr, sides[side]) - gold(fr, sides[300 - side])
                    rows.append({
                        "GameId": g["GameId"], "team": names[side], "objective": inst["kind"], "instance": k,
                        "window": label, "taker": side in inst["takers"], "gold_diff": gd,
                        "gold_state": "leading" if gd >= GOLD_LEAD else "trailing" if gd <= -GOLD_LEAD else "even",
                        **v,
                    })

        # (b) spawn-anchored first dragon: frames at spawn-120 s, -60 s, spawn
        drag = [i for i in insts if i["kind"] == "DRAGON"]
        pit = (drag[0]["x"], drag[0]["y"]) if drag else (9866, 4414)
        for off in (-120, -60, 0):
            fi = (FIRST_DRAGON_SPAWN_MS + off * 1000) // 60_000
            fr = frames[fi]
            dead = dead_at(fr["timestamp"], deaths)
            for side in (100, 200):
                v = team_view(fr, sides[side], pit, dead)
                spawn_rows.append({"GameId": g["GameId"], "team": names[side], "side": side, "offset_s": off,
                                   "took_first_dragon": bool(drag) and side in drag[0]["takers"], **v})

    df, sp = pd.DataFrame(rows), pd.DataFrame(spawn_rows)
    df.to_csv(OUT / "kill_anchored_team_setups.csv", index=False)
    sp.to_csv(OUT / "first_dragon_spawn_setups.csv", index=False)

    fk = pd.DataFrame([(k, len(v), min(v), np.median(v)) for k, v in first_kill.items()],
                      columns=["objective", "games", "earliest_first_kill_s", "median_first_kill_s"])
    fk.to_csv(OUT / "first_kill_times.csv", index=False)

    near = (df[df.window == "0-60s before"].groupby(["objective", "taker"])["near"]
              .agg(["mean", "count"]).round(2).unstack("taker"))
    near.to_csv(OUT / "players_near_pit_taker_vs_not.csv")
    contest = (df[(df.window == "0-60s before") & (~df.taker)]
               .assign(contest=lambda d: d.near >= 3)
               .groupby(["objective", "gold_state"])["contest"].agg(["mean", "count"]).round(3).unstack("gold_state"))
    contest.to_csv(OUT / "non_taker_contest_by_gold_state.csv")
    spawn_tab = sp.groupby(["offset_s", "took_first_dragon"])["near"].agg(["mean", "count"]).round(2)
    spawn_tab.to_csv(OUT / "first_dragon_near_by_offset.csv")

    print(f"kill-anchored team-setups: {len(df)} rows ({df[['GameId','instance']].drop_duplicates().shape[0]} objective instances)")
    print(fk.sort_values("median_first_kill_s").to_string(index=False))
    print("\nmean players within 3000 of pit, 0-60 s before the kill (taker vs non-taker):")
    print(near.to_string())
    print("\nnon-taking team: share with >=3 players near the pit (contesting), by gold state:")
    print(contest.to_string())
    print("\nfirst dragon, spawn-anchored: mean players near the dragon pit")
    print(spawn_tab.to_string())


if __name__ == "__main__":
    main()
