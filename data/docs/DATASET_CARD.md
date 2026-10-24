# Dataset card (Datasheets for Datasets structure)

**Name (working):** *Setups Before Spawns* — tracked positions and team set-ups before neutral-objective spawns in professional League of Legends (MSI 2025, Worlds 2025).
**Snapshot:** 2026-10-22/23. **Companion paper:** "Allocation Before the Clock" (arXiv version). See the repository `README.md` for terms and attributions and `LAYOUT.md` for schemas.

## 1. Motivation
- **Purpose.** Study how teams position before neutral objectives spawn (Atakhan, Baron, Dragon, Elder, Grubs/Horde, Herald), a question the public per-minute data cannot answer because positions are sampled once every 60 s.
  The dataset provides 2-frames-per-second champion tracks reconstructed from the official broadcast minimap, plus the derived set-up, fight and value tables behind every number in the paper.
- **Gap filled.** Riot's V5 timelines give exact positions only each minute. The tracker (LDA detector + Viterbi on a navigation graph, anchored on V5 positions) fills the gaps; validation against held-out V5 minutes is released with it.
- **Who/funding.** UNSW Sydney research project. No external funding.

## 2. Composition
| Item | Count |
|---|---|
| Core games (tracked) | **164** = MSI 2025 80 (19 series, 10 teams) + Worlds 2025 84 (Main Event 80 games / 40 series + Play-In 4 games / 1 series); overall 60 series, 19 teams |
| Neutral-objective spawns (core) | **1,516** (Atakhan 164, Baron 224, Dragon 773, Elder 27, Grubs 164, Herald 164); median spawn→kill delay: grubs 21 s, dragon 49 s, herald 50 s, Atakhan 109 s, Baron 117 s |
| Team set-up snapshots | 56,220 team-snapshots = 2,994 team-setups (1,497 spawn×2 teams) × 21 time offsets (−120…+180 s, step 15 s); 164 games |
| Minimap tracks | 5,758,940 rows (164 games × 10 champions × ~0.5 s); state shares: detected 76.8 %, interpolated 17.8 %, dead 4.1 %, pinned to V5 0.8 %, held 0.4 % |
| Fights | 2,325 fights (164 games; Tullock/Lanchester fits) |
| Clock alignments | 166 games (164 core + 2 Worlds 2026 contrast), 460 segments (437 live, 23 replay), 0 unmatched |
| Minimap calibrations | 164 games, 4-parameter orthographic fit each |
| Navigation graph | 12,944 walkable 100-unit cells (150×150 grid, ≥3 games), 99.2 % in one component, built from 1,994 games / 407k base-terrain positions |
| Game/series metadata | `games_all.csv` 2,853 pro games 2023–2026 (LCK + international: 685/679/754/735 per year); VOD manifest 218 rows (core 164, contrast 54: T1 2023/24 32, Worlds 2026 22) |
| Context (not released raw) | V5 timelines 2,679 games; Oracle's Elixir 2023–2026 (493k rows); Leaguepedia tables; PandaSkill 37,459 games |
- **Not included:** video, frames, audio; raw V5; raw Oracle's Elixir; raw PandaSkill; player names/ids (roles + team only).
- **Sample.** Not a random sample: all MSI 2025 and Worlds 2025 games with an available YouTube VOD, under a single objective rule set (patch ≥ 25.09 style: 3 grubs per camp, Atakhan). The 54 contrast games (other years/rule sets) have manifest rows only; they are **not tracked** (A9 not available).
- **Labels/targets.** None; measured quantities. Ground truth for validation: V5 positions at odd minutes (23,286 champion-minutes).
- **Errors/noise.** Tracker vs V5 at held-out minutes: pooled median error **177 units** (map width 14,820), p90 3,739, p95 7,582; share <300 / <600 / <1000 units = 0.697 / 0.785 / 0.817. Detected (78 %) median 146; interpolated (21 %) median 921; held median 3,012.
  About 7 % of detections are gross identity errors; formation exact agreement with V5 is only 0.536 (Rand 0.848). Use `state` and `q_*` columns to filter (dropping >50 % held/interpolated snapshots keeps 104 of 166 headline contrasts).
- **Sensitive data.** No personal data beyond professional team names (public). No player handles. Leaguepedia's `MVP` column dropped.

## 3. Collection process
- **Leaguepedia:** `Special:CargoExport` and MediaWiki API (tables MatchSchedule, MatchScheduleGame, PicksAndBansS7, ScoreboardGames/Players; V5 pages), ~2 s/request, resumable (`code/scripts/fetch_leaguepedia.py`, `fetch_v5.py`). No Cloudflare bypass.
- **V5 timelines/postgame:** Leaguepedia "V5 data" namespace (Riot match-v5 format), 2,679 games.
- **Oracle's Elixir / PandaSkill:** downloaded from their public hosts for cross-checks (not bundled).
- **Video:** official LoL Esports YouTube VODs, only the game segment, video only, 1080p, via yt-dlp (`code/scripts/download_vods.py`); 214 files / 131 GB downloaded locally, never released.
- **Time:** collection snapshot 2026-10-22; Worlds 2026 contrast games are from October 2026.

## 4. Preprocessing / cleaning / labelling
1. **Spawn table** (`build_objective_spawns.py`): spawn time from fixed schedule/rules, kill time/position/team from V5 events.
2. **Clock alignment** (`clock.py`, `run_clock.py`): broadcast clock cropped at 10 fps; own digit recogniser trained from one hand-checked reading; each second-boundary gives a (VOD time, game time) pair → piecewise offset handling pauses/replays. Live-frame share median 80.6 % (MSI) / 79.2 % (Worlds); misread median 0.00 % / 0.4 %.
3. **Minimap registration** (`minimap.py`): per game, median minimap image vs navigation-graph walkable mask, 4 parameters `(ax, sx, ay, sy)`: `px = ax + sx·x`, `py = ay − sy·y` on the 276×276 crop. Checks: 28 fixed buildings land on their icons; V5 positions land on icons; two crops differ <0.04 px.
4. **Tracker** (`tracker.py`, `run_tracker.py`): per-champion LDA template detector, Viterbi over candidates {peak, hold, dead} with geodesic movement cost; anchored on V5 per-minute positions; validation run uses even-minute anchors, evaluates odd minutes.
5. **Set-ups** (`setups_v.py`, `setups.py`): at each offset, team formation partition (agglomeration at geodesic radius), zone labels relative to the pit, arrival-time order statistics `tau1/tau3/tau5`, pit control, players within the pit radius (1,500 geodesic units; robustness 1,000/2,000 in the code).
6. **Fights, values, fits:** stochastic-Lanchester S7 test and Tullock fight law; win-probability surface W (cross-fitted); event values (WPA); HASSE (K = 1…8, reported at K = 6) and LB-QRE fits with series-block bootstrap. Results summarised in `results/` and the paper.
- **Raw data preserved?** Yes, locally; not released. All scripts released.
- **Known data issues** : 5 games with long pauses had truncated clips (re-download planned); 3 games with changed broadcast layout lack clock labels for the first 4–17 min; Worlds main R2_7_1 VOD starts at minute 17; MSI R4_1_2 VOD is choppy (37 % usable); Worlds 2023 quarter-final scoreboards missing; MSI 2024 lacks RiotPlatformGameId.

## 5. Uses
- **Intended:** analysis of pre-objective positioning; tracker/formation benchmarks; discrete-game / team-behaviour modelling; reproducing the paper's tables and figures.
- **Not suited for:** individual player evaluation or scouting (no player ids, tracker error), betting, claims about other patches/rule sets, anything requiring video.
- **Caveats for users:** the HASSE model selects no finite K ≤ 8; team fingerprints are mostly noise at 5–394 setups per team; the QRE channel decomposition reproduces only about a quarter of the observed commitment gap (a negative result reported in the paper).
- **Prohibited by us (request):** de-anonymisation of players, commercial exploitation of Riot-owned game data; Riot's legal notice applies.

## 6. Distribution
- **Hosting:** this GitHub repository. Size ≈ **87 MB** default build (tracks 70 MB). The optional 188 MB geodesic distance matrix is not included; rebuild it from `navgraph/nav_nodes.csv` with `code/lolsetup/navgraph.py`.
- **Licence:** data CC BY-SA 4.0; code MIT; video not distributed (see `README.md`).
- **Third-party restrictions:** Leaguepedia attribution + ShareAlike; Riot disclaimer; no raw Riot/OE/PandaSkill/YouTube material.
- **Export controls / regulatory:** none known.
- **DOI/citation:** none yet; BibTeX in `README.md`.

## 7. Maintenance
- **Maintainer:** paper's corresponding author (zheng.gao1@unsw.edu.au).
- **Updates:** versioned releases (v1.0 = arXiv v1); errata via GitHub issues; Worlds 2026 contrast games would be added as a separate version if tracked.
- **Takedown:** the release can be withdrawn/changed on request from Riot, Leaguepedia or other rights holders; .
- **Extension:** reproduction scripts allow rebuilding from public sources; contributions via pull request.
