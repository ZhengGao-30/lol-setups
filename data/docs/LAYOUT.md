# Release layout and table schemas

Release layout (`data/` in this repository). Default build ≈ **75 MB**.
Conventions: coordinates are Riot engine units (map 0…~14,870; blue base bottom-left, red top-right); `game_s` = game-clock seconds (not video time);
`side` 100 = blue, 200 = red; roles `T` top, `J` jungle, `M` mid, `B` bottom/ADC, `S` support; pid 1–5 = blue T,J,M,B,S, 6–10 = red T,J,M,B,S (from `io.TEAM_IDS`/`ROLE_SHORT`; **verify pid order once before release**).
Primary key linking everything: `RiotPlatformGameId` (e.g. `LOLTMNT03_260300`) = Oracle's Elixir `gameid` = V5 page name; `GameId` = Leaguepedia game id.

```
data/
  docs/                     DATASET_CARD.md LAYOUT.md
  manifests/                core_worlds2025.csv core_msi2025.csv holdout_worlds2026.csv t1_appendix.csv games_all.csv
                            objective_spawns.csv vod_manifest.csv v5_targets.csv v5_targets_priority.csv building_positions.csv summary.csv
  alignment/                clock_games.csv clock_segments.csv clock_flagged.json
  minimap/                  minimap_calibration.csv
  tracks/                   minimap_tracks_2fps.parquet                 (~57 MB, 5.76 M rows)
  navgraph/                 nav_nodes.csv
  derived/                  team_setup_snapshots.parquet fights.parquet event_values.csv arrival_race.csv
                            hasse_posteriors_K6.parquet hasse_patterns_K6.json
  results/                  tracker/ v_layer/ data/ fights/ context/ baseline/ robustness/ t1/ figures/ winsurface/
                            hasse_v/ fingerprints_K5|K6|K8/ qre/        (summary tables and figures behind the paper)
```
Excluded on purpose: `data/video/*`, `data/raw/*`, `fits/*.pkl` (pickle; QRE numbers are in `results/qre`, `results/robustness`), per-game tracker eval files (308 small files; pooled in `results/tracker/eval_pooled.parquet`), `results/qre_spec_history` (development), older `hasse*` dirs.

## manifests/
**`core_worlds2025.csv` (84 rows), `core_msi2025.csv` (80), `holdout_worlds2026.csv` (116), `t1_appendix.csv` (177), `games_all.csv` (2,853)** — one row per game; Leaguepedia `MatchScheduleGame` + `PicksAndBansS7` + scoreboard fields. `MVP` removed; VOD URLs without tracking parameters.
| column | type | meaning |
|---|---|---|
| OverviewPage, GameId, MatchId, N_GameInMatch | str, str, str, int | tournament page; Leaguepedia game id; series id; game number in series |
| Blue, Red | str | team names; Winner = 1 (blue)/2 (red) (float) ; FirstPick, Selection, PickSelection | str | draft side/priority fields |
| IsChronobreak, IsRemake, FF | bool/str | flags |
| Vod, VodPB, VodGameStart, VodPostgame | url | YouTube links; `?t=` = seconds into video (VodGameStart = game start, VodPB = pick/ban) |
| RiotPlatformGameId, RiotPlatformId, RiotGameId | str | Riot ids (join key) |
| BestOf, SeriesStartUTC, SeriesPatch, Tab, Round, Phase | mixed | series info |
| Team1, Team2, Team1Score, Team2Score, WinTeam, LossTeam | | series-level teams/scores/winner |
| DateTime_UTC, Gamelength (mm:ss), Gamelength_Number (min), Patch | | game date, length, patch |
| Team1Picks, Team2Picks, Team1Bans, Team2Bans | str (python list) | 5 picks/bans each |
| HasOrderedDraft, Played, IsInternational, Year, FearlessEra, HasScoreboard | bool/float | derived flags |

**`objective_spawns.csv` (1,516)** — one row per neutral-objective spawn, core games.
| column | type | unit/meaning |
|---|---|---|
| GameId, RiotPlatformGameId, event, Blue, Red | str | event = "MSI 2025" / "Worlds 2025" |
| objective | str | ATAKHAN, BARON_NASHOR, DRAGON, ELDER, HORDE (grubs), RIFTHERALD |
| idx | int | n-th spawn of that objective in the game |
| spawn_s, kill_s, game_end_s | float s | game clock; kill_s NaN if never killed |
| taken_by | float | 100/200 team that took it |
| kill_x, kill_y | float | position of the kill (units) |
| grubs_100, grubs_200 | float | grubs taken by each side (HORDE rows) |
| camp_clear_s, latency_s | float s | camp clear time; latency = kill_s − spawn_s |

**`vod_manifest.csv` (218)** — video *pointers*: GameId, OverviewPage, Blue, Red, N_GameInMatch, RiotPlatformGameId, `youtube_id` (11 chars), `start_s`, `end_s` (VOD seconds of the game segment), `gamelength_s`, `vod_game_start_url`, `vod_pb_url`, `set` (core_worlds2025 84, core_msi2025 80, contrast_t1_2023_2024 32, contrast_worlds2026 22).
**`building_positions.csv` (26)** — team, b (building type), lane, tower, n, x, y, sx, sy, nuniq: fixed structure coordinates used for calibration checks.
**`v5_targets*.csv`** — GameId, OverviewPage, RiotPlatformGameId: list of games whose V5 pages users must fetch (2,679 / 234 priority).
**`summary.csv`** — per tournament counts (series, game_rows, played, scoreboards, ordered_drafts, vod_game_start, riot_ids).

## alignment/
**`clock_games.csv` (166)**: name (video file stem), n_frames, frac_readable, frac_live, misread_rate, n_live, n_replay, first_live_s, last_live_s (game seconds), GameId, RiotPlatformGameId, set, gamelength_s, end_diff_s (last live − game length).
**`clock_segments.csv` (460)**: name, vod_start_s, vod_end_s (video seconds), offset_s (game_s = video_s + offset_s; checked on the first MSI game: 2.7 + 28.15 = 30.85 = first_live_s), n_boundaries (second-boundary matches), mad_s (offset scatter), live (bool; False = replay).
**`clock_flagged.json`**: list of game names with alignment issues.

## minimap/
**`minimap_calibration.csv` (164)**: name, RiotPlatformGameId, ax, sx, ay, sy (`px = ax + sx·x`, `py = ay − sy·y`, pixels on the 276×276 minimap crop; x,y in units), register_corr (correlation with walkable mask), live_frames, frames.

## tracks/minimap_tracks_2fps.parquet (5,758,940 rows)
| column | type | meaning |
|---|---|---|
| RiotPlatformGameId | str | game |
| k | int16 | frame index of the 2-fps crop (video-time based; `game_s` is authoritative) |
| game_s | float32 s | game clock (0.5 s steps) |
| pid | int16 | participant 1–10 (see conventions) |
| x, y | float32 units | tracked position |
| logit | float32 | calibrated detector log-odds at the chosen position (−/low for held/interpolated) |
| state | category | `det` detected 76.8 %, `interp` 17.8 %, `dead` 4.1 %, `pin` (V5 anchor) 0.8 %, `hold` (occluded, kept) 0.4 % |
Dropped relative to the working files: `px, py` (detector pixel peak; recompute with `minimap_calibration`).

## navgraph/nav_nodes.csv (12,944)
node (0…12,943; index used by `node_*` columns), cell_i, cell_j (grid index, 100-unit cells), x_min, y_min (= 100·cell), games_seen (distinct games with a position in the cell), position_count. Edges: 8-neighbour + knight steps, no corner cutting, weight = 100·hypot (rebuild with `lolsetup/navgraph.py`; or use the optional distance matrix `dist` [12,944² uint16, units]).

## derived/team_setup_snapshots.parquet (56,220 rows × 68)
One row = one team at one offset around one spawn.
| columns | meaning |
|---|---|
| RiotPlatformGameId, GameId, event, objective, idx, pit | spawn identity; pit = dragon / baron / atakhan |
| spawn_s, offset_s, game_s | spawn time; offset −120…+180 step 15; game time of the matched frame (s) |
| side, team, opponent | 100/200; team names |
| taken_by_team | bool/NaN: did this team take the objective |
| kill_s, obj_up | kill time; objective still alive at this snapshot |
| gold_diff | team − opponent gold (from V5 frames, interpolated) |
| grub_share | share of 3 grubs taken by team (HORDE only) |
| rift_element_pre | dragon element known before the snapshot (soul reveal) |
| pit_control | share of pit-area cells the team reaches first (0–1) |
| x_R, y_R, alive_R, dpit_R, node_R, zone_R (R ∈ T,J,M,B,S) | per-role position (NaN if dead), alive flag, geodesic distance to pit centre (units), nav node, zone label ∈ {pit, own_pitside, opp_pitside, pit_lane, mid, far, base} |
| alive | number alive |
| partition, block_zones, shape, n_blocks, merge_heights | formation: blocks of roles e.g. `BS|T|J|M`; zone of each block; sizes `2+1+1+1`; merge distances (units) |
| k_pit | alive players within 1,500 geodesic units of the pit |
| tau1, tau3, tau5 | seconds for the 1st/3rd/5th fastest player to reach the pit (geodesic distance / speed; inf if none) |
| q_det, q_hold, q_interp, q_pin | shares of the team's alive players tracked in each state (quality filter) |
| q_logit | mean detector log-odds |
| opp_alive, opp_k_pit, opp_tau1, opp_tau3 | same for opponent |

## derived/fights.parquet (2,325)
RiotPlatformGameId; t0 (s, fight start); x, y (units, centroid); n_kills; kills_blue, kills_red; n_blue, n_red (participants incl. V5 killer/victim/assist sets); n_blue_tracked, n_red_tracked (tracked within 2,000 units 1 s before kill); gold_pp_blue, gold_pp_red (mean gold per player); winner (100/200).
## derived/event_values.csv (192)
kind (`wpa` = 3-minute win-probability added; `result` = total effect), type (ATAKHAN, BARON_NASHOR, DRAGON_1…4, ELDER, FIGHT, FIGHT_OBJ, HORDE, RIFTHERALD, TOWER), minute, n (events), `value_at_W=0.5 (points)` (win-prob percentage points), a (coefficient on logit W).
## derived/arrival_race.csv (1,493)
RiotPlatformGameId, group (objective), MatchId, first3_blue, first3_red (s at which a side first has 3 at the pit; inf = never), race (blue/red/none), outcome (who took it).
## derived/hasse_posteriors_K6.parquet (2,994) / hasse_patterns_K6.json
Per team-setup: p_pattern0…5 (posterior at the spawn), occ_patternK / s_patternK (occupancy share and spawn-time weight), RiotPlatformGameId, objective, idx, side. Patterns JSON = dictionary of the K = 6 patterns (centre paths, rigidity θ, inertia ι).

## results/
CSV/JSON/MD summaries, one folder per analysis. Key schemas: `tracker/summary_*.csv` (quantiles q25…q99, n, mean, lt300/lt600/lt1000), `tracker/eval_pooled.parquet` (name, minute, t_ms, pid, x, y [V5 truth], alive, level, k, tx, ty [tracked], state, err), `fights/fight_laws.csv` (model, counts, param, est, lo, hi, n), `qre/team_parameters.csv` (lam, nu, kappa, psi, omega_k, omega_m, lam_obj, lam_rest, contests), `hasse_v/selection.csv` (K, held-out log-lik, BIC, ICL), `data/table1_*.csv`.
