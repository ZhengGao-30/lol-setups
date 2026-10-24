"""Fetch Leaguepedia Cargo tables (schedules, VOD links, scoreboards, ordered drafts).

Uses the logged-out Special:CargoExport endpoint (api.php cargoquery is rate-limited
for anonymous users). Paginates with offset, sleeps between calls, retries politely.
Leaguepedia content is CC BY-SA 3.0 -- attribute and share-alike any redistribution.

Usage:
    python scripts/fetch_leaguepedia.py            # LCK + internationals 2023-2026, all tables
    python scripts/fetch_leaguepedia.py --lpl-lec  # also LPL + LEC 2023-2026 (draft/schedule tables only)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "leaguepedia"
URL = "https://lol.fandom.com/wiki/Special:CargoExport"
UA = "LoLGraphTheoryResearch/0.1 (academic research; polite crawler)"
SLEEP = 2.0
PAGE = 500

INTERNATIONAL = [
    "2023 Season World Championship/Play-In",
    "2023 Season World Championship/Main Event",
    "2024 Season World Championship/Play-In",
    "2024 Season World Championship/Main Event",
    "2025 Season World Championship/Play-In",
    "2025 Season World Championship/Main Event",
    "2026 Season World Championship/Play-In",
    "2026 Season World Championship/Main Event",
    "2023 Mid-Season Invitational",
    "2024 Mid-Season Invitational",
    "2025 Mid-Season Invitational",
    "2026 Mid-Season Invitational",
    "2025 First Stand",
    "2026 First Stand",
]

TABLES = {
    "MatchSchedule": ("MatchId", [
        "OverviewPage", "MatchId", "Team1", "Team2", "Winner", "Team1Score", "Team2Score",
        "BestOf", "DateTime_UTC", "Patch", "Round", "Phase", "Tab", "IsTiebreaker",
        "DisabledChampions", "GridSeriesId",
    ]),
    "MatchScheduleGame": ("GameId", [
        "OverviewPage", "GameId", "MatchId", "N_GameInMatch", "Blue", "Red", "Winner",
        "FirstPick", "Selection", "PickSelection", "IsChronobreak", "IsRemake", "FF",
        "Vod", "VodPB", "VodGameStart", "VodPostgame", "RiotPlatformGameId",
        "RiotPlatformId", "RiotGameId", "MVP",
    ]),
    "ScoreboardGames": ("GameId", [
        "OverviewPage", "Tournament", "GameId", "MatchId", "N_GameInMatch", "Team1", "Team2",
        "WinTeam", "LossTeam", "DateTime_UTC", "Gamelength", "Gamelength_Number", "Patch",
        "Team1Picks", "Team2Picks", "Team1Bans", "Team2Bans", "Team1Players", "Team2Players",
        "Team1Dragons", "Team2Dragons", "Team1Barons", "Team2Barons", "Team1Towers",
        "Team2Towers", "Team1Gold", "Team2Gold", "Team1Kills", "Team2Kills",
        "Team1RiftHeralds", "Team2RiftHeralds", "Team1VoidGrubs", "Team2VoidGrubs",
        "Team1Atakhans", "Team2Atakhans", "Team1Inhibitors", "Team2Inhibitors",
        "RiotPlatformGameId",
    ]),
    "ScoreboardPlayers": ("GameRoleId", [
        "OverviewPage", "GameId", "MatchId", "GameRoleId", "Link", "Name", "Team", "TeamVs",
        "Role", "IngameRole", "Side", "Champion", "Kills", "Deaths", "Assists", "Gold", "CS",
        "DamageToChampions", "VisionScore", "Items", "Trinket", "SummonerSpells",
        "KeystoneRune", "PrimaryTree", "SecondaryTree", "TeamKills", "TeamGold", "PlayerWin",
        "DateTime_UTC",
    ]),
    "PicksAndBansS7": ("GameId", [
        "OverviewPage", "GameId", "MatchId", "N_GameInMatch", "Team1", "Team2", "Winner",
        "Team1Score", "Team2Score", "Phase", "Tab",
        *[f"Team{t}Ban{i}" for t in (1, 2) for i in range(1, 6)],
        *[f"Team{t}Pick{i}" for t in (1, 2) for i in range(1, 6)],
        *[f"Team{t}Role{i}" for t in (1, 2) for i in range(1, 6)],
        "Team1PicksByRoleOrder", "Team2PicksByRoleOrder",
    ]),
}


def where_for(set_name):
    intl = " OR ".join(f'{{t}}.OverviewPage="{p}"' for p in INTERNATIONAL)
    lck = " OR ".join(f'{{t}}.OverviewPage LIKE "LCK/{y} Season%"' for y in range(2023, 2027))
    lpl_lec = " OR ".join(
        f'{{t}}.OverviewPage LIKE "{lg}/{y} Season%"' for lg in ("LPL", "LEC") for y in range(2023, 2027)
    )
    return {"lck_intl": f"({intl}) OR ({lck})", "lpl_lec": f"({lpl_lec})"}[set_name]


def get(params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(URL, params=params, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(data["error"])
                return data
            print(f"  HTTP {r.status_code}; retrying", file=sys.stderr)
        except (requests.RequestException, ValueError) as e:
            print(f"  {type(e).__name__}: {e}; retrying", file=sys.stderr)
        time.sleep(SLEEP * (2 ** k))
    raise RuntimeError(f"giving up on {params}")


def fetch_table(table, set_name):
    order_key, fields = TABLES[table]
    where = where_for(set_name).replace("{t}", table)
    rows, offset = [], 0
    while True:
        params = {
            "tables": table,
            "fields": ",".join(f"{table}.{f}" for f in fields),
            "where": where,
            "order by": f"{table}.{order_key}",
            "limit": PAGE,
            "offset": offset,
            "format": "json",
        }
        batch = get(params)
        rows.extend(batch)
        print(f"  {table} [{set_name}] offset={offset} got={len(batch)} total={len(rows)}")
        time.sleep(SLEEP)
        if len(batch) < PAGE:
            break
        offset += PAGE
    # CargoExport renders underscores in field names as spaces
    df = pd.DataFrame(rows).rename(columns=lambda c: c.replace(" ", "_"))
    df = df[~df["OverviewPage"].astype(str).str.contains("Showmatch", na=False)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lpl-lec", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [("lck_intl", t) for t in TABLES]
    if args.lpl_lec:
        jobs += [("lpl_lec", t) for t in ("MatchSchedule", "MatchScheduleGame", "PicksAndBansS7")]

    for set_name, table in jobs:
        dest = OUT / f"{set_name}__{table}.csv"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        print(f"fetching {table} [{set_name}]")
        df = fetch_table(table, set_name)
        df.to_csv(dest, index=False, encoding="utf-8")
        print(f"  -> {dest.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
