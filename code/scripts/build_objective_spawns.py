"""Objective spawn table for the core video set (Worlds 2025 + MSI 2025), from Riot V5 timelines.

Spawn rules in force for patches 25.09-25.20 (25.S1.1 + 25.09 notes; no objective changes in 25.20):
    elemental drake  first 300 s; respawn = previous drake kill + 300 s, until a team claims soul
    elder            soul-claiming drake kill + 360 s; respawn = elder kill + 360 s
    void grubs       480 s, one camp of 3, no respawn (despawn 14:45)
    rift herald      900 s, once (despawn 24:45)
    atakhan          1200 s, once
    baron            1500 s; respawn = baron kill + 360 s
Spawns at or after game end are dropped; spawned-but-untaken objectives are kept as censored.
Sanity check printed at the end: no kill may precede its inferred spawn.

Output: data/manifests/objective_spawns.csv  (one row per spawn instance = one 'setup' for each team)
"""
import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "data" / "raw" / "v5" / "timeline"
CORE = ["core_worlds2025.csv", "core_msi2025.csv"]
FIXED = {"HORDE": 480, "RIFTHERALD": 900, "ATAKHAN": 1200}
DRAKE_FIRST, DRAKE_RESPAWN = 300, 300
ELDER_DELAY, BARON_FIRST, BARON_RESPAWN = 360, 1500, 360


def kills(tl):
    out = []
    for f in tl["frames"]:
        for e in f["events"]:
            if e["type"] != "ELITE_MONSTER_KILL":
                continue
            kind = e["monsterType"]
            if kind == "DRAGON" and e.get("monsterSubType") == "ELDER_DRAGON":
                kind = "ELDER"
            out.append({"kind": kind, "t": e["timestamp"] / 1000, "team": e.get("killerTeamId"),
                        "x": e["position"]["x"], "y": e["position"]["y"]})
    return sorted(out, key=lambda k: k["t"])


def spawns_for_game(tl):
    end = tl["frames"][-1]["timestamp"] / 1000
    ks = kills(tl)
    rows = []

    def add(kind, idx, spawn, kill, taken_by=None, extra=None):
        if spawn >= end:
            return
        rows.append({"objective": kind, "idx": idx, "spawn_s": spawn,
                     "kill_s": kill["t"] if kill else None,
                     "taken_by": taken_by if taken_by is not None else (kill["team"] if kill else None),
                     "kill_x": kill["x"] if kill else None, "kill_y": kill["y"] if kill else None,
                     "game_end_s": end, **(extra or {})})

    # elemental drakes until soul, then elder
    drakes = [k for k in ks if k["kind"] == "DRAGON"]
    count = {100: 0, 200: 0}
    spawn, soul_t = DRAKE_FIRST, None
    for i, k in enumerate(drakes, 1):
        add("DRAGON", i, spawn, k)
        count[k["team"]] = count.get(k["team"], 0) + 1
        if max(count.values()) >= 4:
            soul_t = k["t"]
            break
        spawn = k["t"] + DRAKE_RESPAWN
    else:
        add("DRAGON", len(drakes) + 1, spawn, None)  # last spawned drake never taken
    if soul_t is not None:
        spawn = soul_t + ELDER_DELAY
        elders = [k for k in ks if k["kind"] == "ELDER"]
        for i, k in enumerate(elders, 1):
            add("ELDER", i, spawn, k)
            spawn = k["t"] + ELDER_DELAY
        add("ELDER", len(elders) + 1, spawn, None)

    # one-off objectives. Grubs are a split resource (one camp of 3): kill_s = first grub kill (when the
    # camp is engaged; defines the risk set), ownership = the team with at least 2 of the 3 grubs.
    for kind, t0 in FIXED.items():
        first = next((k for k in ks if k["kind"] == kind), None)
        if kind == "HORDE":
            g = [k for k in ks if k["kind"] == "HORDE"]
            n100 = sum(k["team"] == 100 for k in g)
            n200 = len(g) - n100
            owner = None if not g else (100 if n100 > n200 else 200 if n200 > n100 else None)
            add(kind, 1, t0, first, taken_by=owner,
                extra={"grubs_100": n100, "grubs_200": n200, "camp_clear_s": max(k["t"] for k in g) if g else None})
        else:
            add(kind, 1, t0, first)

    # baron and respawns
    spawn = BARON_FIRST
    barons = [k for k in ks if k["kind"] == "BARON_NASHOR"]
    for i, k in enumerate(barons, 1):
        add("BARON_NASHOR", i, spawn, k)
        spawn = k["t"] + BARON_RESPAWN
    add("BARON_NASHOR", len(barons) + 1, spawn, None)
    return rows


def main():
    frames = []
    for name in CORE:
        m = pd.read_csv(ROOT / "data" / "manifests" / name)
        for _, g in m.iterrows():
            with gzip.open(V5 / f"{g['RiotPlatformGameId']}.json.gz", "rt", encoding="utf-8") as f:
                tl = json.load(f)
            for r in spawns_for_game(tl):
                frames.append({"GameId": g["GameId"], "RiotPlatformGameId": g["RiotPlatformGameId"],
                               "event": "Worlds 2025" if "World" in g["OverviewPage"] else "MSI 2025",
                               "Blue": g["Blue"], "Red": g["Red"], **r})
    df = pd.DataFrame(frames)
    df["latency_s"] = df["kill_s"] - df["spawn_s"]
    df.to_csv(ROOT / "data" / "manifests" / "objective_spawns.csv", index=False)

    neg = (df["latency_s"] < 0).sum()
    print(f"spawn instances: {len(df)} -> team-setups: {2 * len(df)}; kills before inferred spawn: {neg}")
    summ = df.groupby(["event", "objective"]).agg(
        spawns=("spawn_s", "size"), taken=("kill_s", lambda s: s.notna().sum()),
        median_latency_s=("latency_s", "median"), min_latency_s=("latency_s", "min"))
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
