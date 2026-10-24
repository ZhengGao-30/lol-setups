# Allocation Before the Clock — pro League of Legends objective setups

Website: **https://zhenggao-30.github.io/lol-setups/** · Paper (PDF): [paper/main.pdf](paper/main.pdf) · Data: [`data/`](data) · Code: [`code/`](code)

## What this is
Before every neutral objective in League of Legends (dragon, Baron, Atakhan, grubs, Herald), a professional team decides how many players go to the pit,
how many hold a lane, and when to commit. Public data gives each player's position only once a minute, which is too coarse to see this.
We reconstructed player positions twice a second from the official broadcast minimap for all 164 games of MSI 2025 and Worlds 2025,
turned them into team "setups" at 21 moments around each of 1,516 objective spawns, and studied how setups depend on the game state.
This repository holds the tracked positions, the derived tables behind every number in the paper, the code, and the project website.

**Technical summary.** Tracks come from a per-champion LDA template detector with Viterbi decoding over a walkable-cell navigation graph,
anchored on Riot V5 per-minute positions (median error 177 units against held-out V5 minutes, map width 14,820). Setups are formation partitions at a geodesic radius,
zone labels and arrival-time order statistics relative to the pit. On top of them we fit HASSE (a lattice/Hasse-diagram mixture of setup patterns; K = 1…8, reported at K = 6),
stochastic-Lanchester/Tullock fight laws, a cross-fitted win-probability surface with event values, and a logit-QRE model of team choices. See the paper for the findings and negative results.

## Paper
**Allocation Before the Clock: Graph, Set and Game Theory of Professional League of Legends Objective Setups**
— Zheng Gao, Zhenyu Liu, Yifan Yang, Xiaoyu Li (UNSW Sydney). PDF: [`paper/main.pdf`](paper/main.pdf).

Build: `cd paper && latexmk -pdf main.tex` (an IEEE two-column variant is `main_ieee.tex`). All figures, tables and `numbers.tex` are included;
`python make_all.py` regenerates those that can be rebuilt from `data/`. Details and the list of figures that need private fits (shipped as PDF only): [`paper/README.md`](paper/README.md).

## Layout
```
paper/   the paper: main.pdf, LaTeX source, figures, figure/table factory
docs/    website (GitHub Pages source): index.html, assets/ (figures, paper.pdf)
data/    CC BY-SA 4.0 data: manifests, alignment, minimap, navgraph, tracks, derived, results, docs
code/    lolsetup/ (package), scripts/ (data fetching, manifests), tests/
```
Full schemas are in [`data/docs/LAYOUT.md`](data/docs/LAYOUT.md); the dataset card (motivation, composition, errors, intended use) is [`data/docs/DATASET_CARD.md`](data/docs/DATASET_CARD.md).

| Part | Contents |
|---|---|
| `data/manifests/` | game and series metadata, objective spawns, VOD pointers (YouTube id + start/end seconds, no video) |
| `data/alignment/`, `data/minimap/` | broadcast-clock alignment and per-game minimap calibration |
| `data/tracks/minimap_tracks_2fps.parquet` | 5.76 M rows: 164 games × 10 champions × 0.5 s, with a `state` column (detected / interpolated / dead / pinned / held) |
| `data/navgraph/nav_nodes.csv` | 12,944 walkable 100-unit cells |
| `data/derived/` | team setup snapshots (56,220 rows), fights, event values, arrival race, HASSE posteriors and patterns |
| `data/results/` | summary tables and figures behind the paper |

Roles are T/J/M/B/S and teams only; there are no player names or ids. Coordinates are Riot engine units; `game_s` is game-clock time.

## Load
```python
import pandas as pd

tracks = pd.read_parquet("data/tracks/minimap_tracks_2fps.parquet")   # needs pyarrow
setups = pd.read_parquet("data/derived/team_setup_snapshots.parquet")
spawns = pd.read_csv("data/manifests/objective_spawns.csv")

g = tracks[tracks.RiotPlatformGameId == tracks.RiotPlatformGameId.iloc[0]]
print(g[g.game_s == 600][["pid", "x", "y", "state"]])          # ten champions at 10:00
print(setups.groupby("objective").size())
```
Filter on `state` (tracks) or the `q_*` columns (setups) if you need only well-observed positions: about 7 % of detections are gross identity errors.
Videos are not distributed; `data/manifests/vod_manifest.csv` and `code/scripts/download_vods.py` let you fetch the segments yourself.

## Licences
- **Data** (`data/`): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — see [`data/LICENSE`](data/LICENSE).
- **Code and website source** (`code/`, `docs/`): MIT — see [`LICENSE`](LICENSE).
- **Request (not a licence term):** please use the dataset for non-commercial purposes only and respect Riot Games' rights in the underlying game data.

## Attributions and notices
- **Leaguepedia.** Game and series manifests are adapted from Leaguepedia ([lol.fandom.com](https://lol.fandom.com)), licensed CC BY-SA 3.0 ([text](https://creativecommons.org/licenses/by-sa/3.0/)); they are redistributed here under CC BY-SA 4.0 with this attribution.
- **Riot Games.** *This project was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.* Riot Games owns the underlying game data; no raw Riot match JSON is included.
- **Oracle's Elixir** (Tim Sevenhuysen, oracleselixir.com) and **PandaSkill** (PandaScore, [github.com/PandaScore/PandaSkill](https://github.com/PandaScore/PandaSkill)) were used for cross-checks and baselines and are cited in the paper; none of their data is bundled and nothing in `data/` is derived from their columns.
- **Not included:** video, frames, crops, audio, raw Riot V5 timelines, raw third-party tables, player names.

Rights holders who want something removed or changed can write to the contact below.

## Citation
```bibtex
@misc{allocation_before_the_clock,
  title        = {Allocation Before the Clock: Graph, Set and Game Theory of Professional League of Legends Objective Setups},
  author       = {Gao, Zheng and Liu, Zhenyu and Yang, Yifan and Li, Xiaoyu},
  institution  = {UNSW Sydney},
  year         = {2026},
  note         = {Preprint},
  url          = {https://zhenggao-30.github.io/lol-setups/}
}

@dataset{setups_before_spawns,
  title        = {Setups Before Spawns: tracked positions and team set-ups before neutral objectives in professional League of Legends},
  author       = {Gao, Zheng and Liu, Zhenyu and Yang, Yifan and Li, Xiaoyu},
  institution  = {UNSW Sydney},
  year         = {2026},
  version      = {v1.0},
  publisher    = {GitHub},
  url          = {https://github.com/ZhengGao-30/lol-setups}
}
```

## Contact
Zheng Gao, UNSW Sydney — zheng.gao1@unsw.edu.au
