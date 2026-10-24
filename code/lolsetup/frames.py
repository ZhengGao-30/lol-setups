"""Per-frame team state from Riot V5 timelines (outcome layer input).

One row per (game, frame) from the blue team's perspective, with team-level state for both sides.
Only quantities observable at the frame time enter (events with timestamp <= frame time).
BUILDING_KILL.teamId is the team that OWNED the building (verified 1,024/1,024), so a building
counts as destroyed BY the other team. Buff windows: Baron 180 s, Elder 150 s.

Usage:  python -m lolsetup.frames  -> fits/frames_state.parquet
"""
from collections import defaultdict

import pandas as pd

from . import io

BARON_BUFF_S, ELDER_BUFF_S = 180, 150
DRAKES = ("FIRE_DRAGON", "WATER_DRAGON", "EARTH_DRAGON", "AIR_DRAGON", "HEXTECH_DRAGON", "CHEMTECH_DRAGON")


def winner(tl, pg):
    for f in tl["frames"]:
        for e in f["events"]:
            if e["type"] == "GAME_END" and e.get("winningTeam") in (100, 200):
                return e["winningTeam"]
    for t in pg.get("teams", []):
        if t.get("win") is True:
            return t["teamId"]
    return None


def game_frames(gid):
    tl, pg = io.load_v5(gid), io.load_v5(gid, "postgame")
    win = winner(tl, pg)
    events = sorted((e for f in tl["frames"] for e in f["events"]), key=lambda e: e["timestamp"])
    rows, ei = [], 0
    c = {t: defaultdict(float) for t in (100, 200)}
    last = {t: {"baron": -1e9, "elder": -1e9} for t in (100, 200)}
    for f in tl["frames"]:
        ts = f["timestamp"]
        while ei < len(events) and events[ei]["timestamp"] <= ts:
            e = events[ei]
            ei += 1
            typ = e["type"]
            if typ == "CHAMPION_KILL" and e.get("killerId", 0) > 0:
                c[100 if e["killerId"] <= 5 else 200]["kills"] += 1
            elif typ == "BUILDING_KILL" and e.get("teamId") in (100, 200):
                taker = 300 - e["teamId"]
                c[taker]["towers" if e.get("buildingType") == "TOWER_BUILDING" else "inhibs"] += 1
            elif typ == "TURRET_PLATE_DESTROYED" and e.get("teamId") in (100, 200):
                c[300 - e["teamId"]]["plates"] += 1
            elif typ == "ELITE_MONSTER_KILL" and e.get("killerTeamId") in (100, 200):
                t, mt, sub = e["killerTeamId"], e["monsterType"], e.get("monsterSubType")
                if mt == "DRAGON" and sub == "ELDER_DRAGON":
                    last[t]["elder"] = e["timestamp"]
                elif mt == "DRAGON":
                    c[t]["drakes"] += 1
                    if sub in DRAKES:
                        c[t][sub.lower()] += 1
                elif mt == "BARON_NASHOR":
                    last[t]["baron"] = e["timestamp"]
                    c[t]["barons"] += 1
                elif mt == "HORDE":
                    c[t]["grubs"] += 1
                elif mt == "RIFTHERALD":
                    c[t]["heralds"] += 1
                elif mt == "ATAKHAN":
                    c[t]["atakhans"] += 1
            elif typ == "DRAGON_SOUL_GIVEN" and e.get("teamId") in (100, 200):
                c[e["teamId"]]["soul"] = 1
        row = {"RiotPlatformGameId": gid, "frame_ms": ts, "minute": ts / 60000, "blue_win": int(win == 100)}
        for t, ids in io.TEAM_IDS.items():
            p = "b_" if t == 100 else "r_"
            pf = [f["participantFrames"][str(i)] for i in ids]
            row[p + "gold"] = sum(q["totalGold"] for q in pf)
            row[p + "xp"] = sum(q["xp"] for q in pf)
            row[p + "alive"] = sum(q["championStats"]["health"] > 0 for q in pf)
            row[p + "baron_buff"] = int(0 <= ts - last[t]["baron"] <= BARON_BUFF_S * 1000)
            row[p + "elder_buff"] = int(0 <= ts - last[t]["elder"] <= ELDER_BUFF_S * 1000)
            for k in ("kills", "towers", "inhibs", "plates", "drakes", "barons", "grubs", "heralds", "atakhans", "soul",
                      *[d.lower() for d in DRAKES]):
                row[p + k] = c[t][k]
            row[p + "soul_point"] = int(c[t]["drakes"] == 3 and not c[t]["soul"])
        rows.append(row)
    return rows if win is not None else []


def main():
    g = io.archive_games()
    rows = []
    for gid in g["RiotPlatformGameId"]:
        rows.extend(game_frames(gid))
    df = pd.DataFrame(rows).merge(g[["RiotPlatformGameId", "GameId", "MatchId", "OverviewPage", "Year"]],
                                  on="RiotPlatformGameId", how="left")
    out = io.ROOT / "fits" / "frames_state.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} frames from {df['RiotPlatformGameId'].nunique()} games -> {out}")


if __name__ == "__main__":
    main()
