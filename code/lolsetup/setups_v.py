"""V layer: team-setup snapshot paths from the 2-fps minimap tracks, for every objective spawn of the core games.

Same objects as the R layer (setups.snapshot_arrays: formation partition, merge heights, zones, pit distances,
k-arrival order statistics, pit control), on a dense spawn-anchored grid OFFSETS_V = -120..+60 s every 15 s and
for all spawns (respawned dragons and barons, Elder included), not only the fixed-time first spawns.
The post-spawn part of the grid (to +180 s) is read at the commitment time t_c (qre_data: first snapshot with >= 3 players
of either team at the pit, or the last snapshot before the objective is taken, truncated at +180 s); HASSE / TAGNet
use the -120..+60 / -90..0 sub-grids.
Per snapshot: the live track frame nearest to spawn + offset (within MATCH_S; otherwise the snapshot is not
emitted and counts as missing downstream); positions from the production track (state 'dead' -> not alive);
movement speed from the preceding V5 minute frame; team gold linearly interpolated between V5 minute frames.
Quality columns per team-snapshot: shares of positions that were detected, held (covered icon, no detection)
or pinned (V5 minute), and the mean emission log-odds. Untaken Atakhans are skipped (pit side not observable).
obj_up marks the risk set exactly as in the R layer.

Usage:  python -m lolsetup.setups_v   -> fits/v_layer_setups.parquet (+ fits/v_layer_skipped.json)
"""
import json
import os

import numpy as np
import pandas as pd

from . import geometry, io, setups
from .navgraph import NavGraph
from .run_clock import vod_name

OFFSETS_V = tuple(range(-120, 181, 15))  # pre-spawn -120..-15, spawn, commitment window up to +180 s
MATCH_S = 1.0


def game_rows(nav, pits, man_row, sp_game, tr):
    frames = io.load_v5(man_row.RiotPlatformGameId)["frames"]
    t_ms = np.array([f["timestamp"] for f in frames], float)
    gold = np.array([[sum(f["participantFrames"][str(i)]["totalGold"] for i in io.TEAM_IDS[t]) for t in (100, 200)]
                     for f in frames], float)
    speed = np.array([[max(f["participantFrames"][str(i)]["championStats"]["movementSpeed"], 1) for i in range(1, 11)]
                      for f in frames], float)
    reveal = next(((e["name"], e["timestamp"]) for f in frames for e in f["events"]
                   if e["type"] == "DRAGON_SOUL_GIVEN" and e.get("teamId") == 0), (None, None))
    tr = tr.sort_values(["k", "pid"])
    ks = tr["k"].unique()
    gs = tr.groupby("k")["game_s"].first().to_numpy()
    X = tr["x"].to_numpy().reshape(len(ks), 10)
    Y = tr["y"].to_numpy().reshape(len(ks), 10)
    ST = tr["state"].to_numpy().reshape(len(ks), 10)
    LG = tr["logit"].to_numpy().reshape(len(ks), 10)
    rows, skipped = [], []
    for s in sp_game.itertuples():
        pit = geometry.pit_of(s.objective, s.kill_x, s.kill_y)
        if pit is None:
            skipped.append({"RiotPlatformGameId": s.RiotPlatformGameId, "objective": s.objective, "idx": s.idx,
                            "reason": "untaken Atakhan"})
            continue
        pit_xy = pits.loc[pit, ["x", "y"]].to_numpy(float)
        pit_node = nav.node(*pit_xy)[0]
        pit_cells = np.flatnonzero(nav.dist[pit_node] <= geometry.PIT_R)
        elem_pre = reveal[0] if reveal[1] is not None and reveal[1] < 1000 * (s.spawn_s + OFFSETS_V[0]) else None
        for off in OFFSETS_V:
            t = s.spawn_s + off
            if t < 0 or t > s.game_end_s:
                continue
            j = int(np.argmin(np.abs(gs - t)))
            if abs(gs[j] - t) > MATCH_S:
                continue
            m = max(int(np.searchsorted(t_ms, 1000 * gs[j], side="right")) - 1, 0)
            g_team = {t_: np.interp(1000 * gs[j], t_ms, gold[:, c]) for c, t_ in enumerate((100, 200))}
            alive = ST[j] != "dead"
            snap = {}
            for team in (100, 200):
                ids = np.array(list(io.TEAM_IDS[team])) - 1
                xy = np.column_stack([X[j, ids], Y[j, ids]])
                xy[~alive[ids]] = pits.loc[pit, ["x", "y"]].to_numpy(float)  # placeholder; dead roles are masked
                snap[team] = setups.snapshot_arrays(nav, xy, alive[ids], speed[m, ids], team, pit_xy, pit_node, pit_cells)
                for r_, a_ in zip(io.ROLE_SHORT, alive[ids]):
                    if not a_:  # no corpse position on the minimap: drop the placeholder
                        snap[team].update({f"x_{r_}": np.nan, f"y_{r_}": np.nan, f"dpit_{r_}": np.nan, f"node_{r_}": -1})
                st, lg = ST[j, ids][alive[ids]], LG[j, ids][alive[ids]]
                snap[team]["_q"] = {"q_det": float(np.mean(st == "det")) if len(st) else np.nan,
                                    "q_hold": float(np.mean(st == "hold")) if len(st) else np.nan,
                                    "q_interp": float(np.mean(st == "interp")) if len(st) else np.nan,
                                    "q_pin": float(np.mean(st == "pin")) if len(st) else np.nan,
                                    "q_logit": float(np.mean(lg)) if len(lg) else np.nan}
            for team in (100, 200):
                mine, theirs = snap[team]["_reach"], snap[300 - team]["_reach"]
                rows.append({
                    "RiotPlatformGameId": s.RiotPlatformGameId, "GameId": s.GameId, "event": s.event,
                    "objective": s.objective, "idx": s.idx, "pit": pit, "spawn_s": s.spawn_s, "offset_s": off,
                    "game_s": float(gs[j]), "side": team,
                    "team": s.Blue if team == 100 else s.Red, "opponent": s.Red if team == 100 else s.Blue,
                    "taken_by_team": (s.taken_by == team) if pd.notna(s.taken_by) else np.nan, "kill_s": s.kill_s,
                    "obj_up": bool(pd.isna(s.kill_s) or s.kill_s > gs[j]),
                    "gold_diff": g_team[team] - g_team[300 - team],
                    "grub_share": (getattr(s, f"grubs_{team}") / 3) if s.objective == "HORDE" else np.nan,
                    "rift_element_pre": elem_pre,
                    "pit_control": float(np.mean(mine < theirs) + 0.5 * np.mean(mine == theirs)),
                    **{k2: v for k2, v in snap[team].items() if k2 not in ("_reach", "_q")},
                    **snap[team]["_q"],
                    **{f"opp_{k2}": snap[300 - team][k2] for k2 in ("alive", "k_pit", "tau1", "tau3")},
                })
    return rows, skipped


def main():
    nav = NavGraph()
    sp = io.spawns()
    pits = geometry.pit_centres(sp)
    man = pd.read_csv(io.MANIFESTS / "vod_manifest.csv")
    man = man[man["set"].str.startswith("core")]
    rows, skipped = [], []
    for r in man.itertuples():
        path = io.ROOT / "fits" / "tracks" / f"{vod_name(r.GameId)}.parquet"
        if not path.exists():
            skipped.append({"RiotPlatformGameId": r.RiotPlatformGameId, "reason": "no track"})
            continue
        rr, sk = game_rows(nav, pits, r, sp[sp["RiotPlatformGameId"] == r.RiotPlatformGameId], pd.read_parquet(path))
        rows += rr
        skipped += sk
    df = pd.DataFrame(rows)
    tag = os.environ.get("V_TAG", "")          # e.g. V_TAG=_r2500 with R_GROUP=2500 (formation-radius robustness)
    out = io.ROOT / "fits" / f"v_layer_setups{tag}.parquet"
    df.to_parquet(str(out) + ".tmp", index=False)
    os.replace(str(out) + ".tmp", out)   # atomic: downstream jobs never see a half-written file
    (io.ROOT / "fits" / f"v_layer_skipped{tag}.json").write_text(json.dumps(skipped, indent=1))
    n_set = df.groupby(["RiotPlatformGameId", "objective", "idx", "side"]).ngroups
    cover = df.groupby(["RiotPlatformGameId", "objective", "idx", "side"]).size() / len(OFFSETS_V)
    print(f"{len(df)} team-snapshots, {n_set} team-setups from {df['RiotPlatformGameId'].nunique()} games; "
          f"grid coverage median {cover.median():.2f}; skipped {len(skipped)} -> {out}")
    print(df.groupby("objective")[["q_det", "q_hold", "q_pin"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
