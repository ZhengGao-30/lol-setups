"""Build the study manifests from the Leaguepedia CSVs.

Outputs (data/manifests/):
    games_all.csv              every LCK + international game 2023-2026, joined schedule/scoreboard/draft
    core_worlds2025.csv        core video set, part 1: Worlds 2025 Play-In + Main Event (all teams)
    core_msi2025.csv           core video set, part 2: MSI 2025 (same post-25.09 objective rules)
    holdout_worlds2026.csv     Worlds 2026 (different objective rules: contrast only)
    t1_appendix.csv            every T1 game at Worlds/MSI 2023-2026 + LCK playoffs 2025-2026
    vod_manifest.csv           per-game YouTube id + game-start / end seconds for the video tier
                               (sets: core_worlds2025, core_msi2025; contrast_worlds2026, contrast_t1_2023_2024)
    v5_targets.csv             RiotPlatformGameIds to pull Riot V5 timelines for
    summary.csv                series/game counts per event
"""
import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "leaguepedia"
OUT = ROOT / "data" / "manifests"

W25 = ["2025 Season World Championship/Play-In", "2025 Season World Championship/Main Event"]
MSI25 = ["2025 Mid-Season Invitational"]
W26 = ["2026 Season World Championship/Play-In", "2026 Season World Championship/Main Event"]
INTL_PREFIX = ("2023 Season World", "2024 Season World", "2025 Season World", "2026 Season World",
               "2023 Mid-Season", "2024 Mid-Season", "2025 Mid-Season", "2026 Mid-Season",
               "2025 First Stand", "2026 First Stand")
VOD_BUFFER_S = 90


def youtube_id_and_t(url):
    if not isinstance(url, str) or not url:
        return None, None
    url = html.unescape(url)  # Leaguepedia stores some links with '&amp;t='
    m = re.search(r"(?:youtu\.be/|v=|/live/)([A-Za-z0-9_-]{11})", url)
    t = re.search(r"[?&#]t=(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?(?:&|$)", url)
    secs = None
    if t and any(t.groups()):
        h, mi, s = (int(x) if x else 0 for x in t.groups())
        secs = 3600 * h + 60 * mi + s
    return (m.group(1) if m else None), secs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ms = pd.read_csv(RAW / "lck_intl__MatchSchedule.csv")
    msg = pd.read_csv(RAW / "lck_intl__MatchScheduleGame.csv")
    sg = pd.read_csv(RAW / "lck_intl__ScoreboardGames.csv")
    pb = pd.read_csv(RAW / "lck_intl__PicksAndBansS7.csv")

    series = ms[["MatchId", "BestOf", "DateTime_UTC", "Patch", "Tab", "Round", "Phase",
                 "Team1", "Team2", "Team1Score", "Team2Score"]].rename(
        columns={"DateTime_UTC": "SeriesStartUTC", "Patch": "SeriesPatch"})
    board = sg[["GameId", "WinTeam", "LossTeam", "DateTime_UTC", "Gamelength", "Gamelength_Number",
                "Patch", "Team1Picks", "Team2Picks", "Team1Bans", "Team2Bans"]]
    games = (msg.merge(series, on="MatchId", how="left")
                .merge(board, on="GameId", how="left")
                .merge(pb[["GameId"]].assign(HasOrderedDraft=True), on="GameId", how="left"))
    games["HasOrderedDraft"] = games["HasOrderedDraft"].fillna(False)
    games["Played"] = games["Winner"].notna()
    games["IsInternational"] = games["OverviewPage"].str.startswith(INTL_PREFIX)
    games["Year"] = games["OverviewPage"].str.extract(r"(20\d\d)").astype(float)
    # Hard fearless in tier-1 play from 2025 (LCK Cup 2025 onward); Worlds Swiss Bo1s draft normally
    games["FearlessEra"] = games["Year"] >= 2025
    games["HasScoreboard"] = games["Gamelength_Number"].notna()
    games = games.sort_values(["SeriesStartUTC", "MatchId", "N_GameInMatch"])
    games.to_csv(OUT / "games_all.csv", index=False)

    played = games[games["Played"]]
    played[played["OverviewPage"].isin(W25)].to_csv(OUT / "core_worlds2025.csv", index=False)
    played[played["OverviewPage"].isin(MSI25)].to_csv(OUT / "core_msi2025.csv", index=False)
    games[games["OverviewPage"].isin(W26)].to_csv(OUT / "holdout_worlds2026.csv", index=False)

    is_t1 = (played["Blue"] == "T1") | (played["Red"] == "T1")
    lck_playoffs = played["OverviewPage"].str.contains(r"LCK/202[56] Season/Season Playoffs", regex=True)
    t1 = played[is_t1 & (played["IsInternational"] | lck_playoffs)]
    t1.to_csv(OUT / "t1_appendix.csv", index=False)

    # Video tier: core = Worlds 2025 + MSI 2025 (one objective-rule regime, post-25.09);
    # contrast = Worlds 2026 Play-In and T1 at Worlds 2023-2024 (different objective rules, descriptive only)
    t1_worlds_early = t1[t1["OverviewPage"].str.match(r"202[34] Season World Championship")]
    video = pd.concat([played[played["OverviewPage"].isin(W25 + MSI25 + W26)], t1_worlds_early]).drop_duplicates("GameId")
    rows = []
    for _, g in video.iterrows():
        vid, t = youtube_id_and_t(g["VodGameStart"])
        if vid is None:
            vid, _ = youtube_id_and_t(g["Vod"])
        length_s = int(round(g["Gamelength_Number"] * 60)) if pd.notna(g["Gamelength_Number"]) else None
        end = t + length_s + VOD_BUFFER_S if (t is not None and length_s) else None
        rows.append({
            "GameId": g["GameId"], "OverviewPage": g["OverviewPage"], "Blue": g["Blue"], "Red": g["Red"],
            "N_GameInMatch": g["N_GameInMatch"], "RiotPlatformGameId": g["RiotPlatformGameId"],
            "youtube_id": vid, "start_s": t, "end_s": end, "gamelength_s": length_s,
            "vod_game_start_url": g["VodGameStart"], "vod_pb_url": g["VodPB"],
            "set": ("core_worlds2025" if g["OverviewPage"] in W25 else
                    "core_msi2025" if g["OverviewPage"] in MSI25 else
                    "contrast_worlds2026" if g["OverviewPage"] in W26 else "contrast_t1_2023_2024"),
        })
    vod = pd.DataFrame(rows)
    vod.to_csv(OUT / "vod_manifest.csv", index=False)

    played[["GameId", "OverviewPage", "RiotPlatformGameId"]].dropna(subset=["RiotPlatformGameId"]) \
        .to_csv(OUT / "v5_targets.csv", index=False)

    summ = (games.groupby("OverviewPage")
                 .agg(series=("MatchId", "nunique"), game_rows=("GameId", "size"), played=("Played", "sum"),
                      scoreboards=("HasScoreboard", "sum"), ordered_drafts=("HasOrderedDraft", "sum"),
                      vod_game_start=("VodGameStart", lambda s: s.notna().sum()),
                      riot_ids=("RiotPlatformGameId", lambda s: s.notna().sum()))
                 .reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)

    print("all games:", len(games), "played:", len(played))
    for name, pages in (("core Worlds 2025", W25), ("core MSI 2025", MSI25)):
        sel = played[played["OverviewPage"].isin(pages)]
        print(f"{name}: {len(sel)} games, {sel['MatchId'].nunique()} series")
    print("hold-out rows (Worlds 2026):", len(games[games["OverviewPage"].isin(W26)]),
          "played:", played["OverviewPage"].isin(W26).sum())
    print("T1 appendix:", len(t1), "games,", t1["MatchId"].nunique(), "series")
    print("video tier:", len(vod), "games; with start ts:", vod["start_s"].notna().sum(),
          "; total hours:", round(vod["gamelength_s"].sum() / 3600, 1))
    print("V5 targets:", played["RiotPlatformGameId"].notna().sum())


if __name__ == "__main__":
    main()
