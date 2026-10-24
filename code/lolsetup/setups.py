"""R layer: team-setup snapshots from exact Riot V5 frames at fixed-time spawns.

For every fixed-time first spawn s in {5:00 dragon, 8:00 grubs, 15:00 herald, 20:00 Atakhan, 25:00 baron}
the engine frames fall on s-60, s and s+60 (frame k at k*60 s, within 1 s). A spawn is kept only if all
three frames are on that grid and none is the game-end frame (whole spawn dropped otherwise; logged).
For each team we record, per snapshot and from positions only:
    - alive flag: health == 0, OR still inside the respawn timer of its last death and within 1,500 u of
      the death spot (Riot frames can report a small residual HP for corpses, which stay at the kill spot)
    - formation: set partition of the alive roles from the geodesic proximity graph (d_G <= R_GROUP)
    - single-linkage merge heights (the whole radius chain), so any radius can be read off later
    - zone of every role (dead -> 'dead') and of every block (zone of the block's geodesic medoid)
    - geodesic distance of every role to the pit centre (k at any radius is recomputed downstream)
    - k-arrival order statistics tau^(k) = k-th smallest d_G(player, pit) / movementSpeed (inf if < k alive)
    - geodesic Voronoi pit control share (pit cells within PIT_R owned by the faster team)
Context (gold, side, revealed rift element, ...) is stored in separate columns and never enters these
objects. obj_up flags whether the objective is still alive at the snapshot (risk set for post-spawn use).

Usage:  python -m lolsetup.setups   -> fits/r_layer_setups.parquet (+ excluded list)
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from . import geometry, io
from .lattice import canon
from .navgraph import NavGraph

R_GROUP = float(os.environ.get("R_GROUP", 1900.0))  # ~5 s of movement at the median speed (381 u/s); radius chain stored too
FIXED = {"DRAGON": 300, "HORDE": 480, "RIFTHERALD": 900, "ATAKHAN": 1200, "BARON_NASHOR": 1500}
OFFSETS = (-60, 0, 60)
GRID_TOL_MS = 1000
CORPSE_R = 1500.0
BRW = np.array([10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5])


def respawn_ms(level, t_kill_ms):
    """Approximate death timer: base respawn wait by level, scaled up after 15:00."""
    t = t_kill_ms / 1000
    tif = 0.0 if t < 900 else min(0.5, 0.00425 * ((t - 900) // 30))
    return 1000 * BRW[min(max(int(level), 1), 18) - 1] * (1 + tif)


def partition(D, alive, r):
    """Connected components of {i ~ j : D_ij <= r} over alive roles -> canonical tuple of blocks."""
    ids = [i for i in range(5) if alive[i]]
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if D[i, j] <= r:
                parent[find(i)] = find(j)
    blocks = {}
    for i in ids:
        blocks.setdefault(find(i), []).append(i)
    return canon(blocks.values())


def label(blocks):
    return "|".join("".join(io.ROLE_SHORT[i] for i in b) for b in blocks)


def merge_heights(D, alive):
    """Single-linkage merge heights among alive roles (= MST edge lengths, zero distances kept)."""
    ids = [i for i in range(5) if alive[i]]
    if len(ids) < 2:
        return []
    h = sorted(linkage(squareform(D[np.ix_(ids, ids)], checks=False), "single")[:, 2].tolist())
    assert len(h) == len(ids) - 1
    return h


def alive_flags(pf, ids, t_ms, deaths):
    """deaths: participantId -> list of (t_ms, x, y), sorted."""
    out = []
    for i, p in zip(ids, pf):
        dead = p["championStats"]["health"] <= 0
        if not dead:
            past = [d for d in deaths.get(i, []) if d[0] <= t_ms]
            if past:
                tk, kx, ky = past[-1]
                near = np.hypot(p["position"]["x"] - kx, p["position"]["y"] - ky) <= CORPSE_R
                dead = near and (t_ms - tk) < respawn_ms(p["level"], tk)
        out.append(not dead)
    return np.array(out)


def team_snapshot(nav, frame, team, pit_xy, pit_node, pit_cells, deaths):
    ids = list(io.TEAM_IDS[team])
    pf = [frame["participantFrames"][str(i)] for i in ids]
    xy = np.array([[p["position"]["x"], p["position"]["y"]] for p in pf], float)
    alive = alive_flags(pf, ids, frame["timestamp"], deaths)
    phantom = int(sum(a == 0 and p["championStats"]["health"] > 0 for a, p in zip(alive, pf)))
    speed = np.array([max(p["championStats"]["movementSpeed"], 1) for p in pf], float)
    return {**snapshot_arrays(nav, xy, alive, speed, team, pit_xy, pit_node, pit_cells), "phantom_dead": phantom}


def snapshot_arrays(nav, xy, alive, speed, team, pit_xy, pit_node, pit_cells):
    """Setup objects of one team from positions (5, 2), alive flags (5,) and movement speeds (5,), role order.
    Shared by the R layer (V5 frames) and the V layer (minimap tracks)."""
    nodes = nav.node(xy[:, 0], xy[:, 1])
    D = nav.d(nodes[:, None], nodes[None, :])
    d_pit = nav.d(nodes, pit_node)
    eta = np.where(alive, d_pit / speed, np.inf)
    blocks = partition(D, alive, R_GROUP)
    zones = geometry.zone(xy[:, 0], xy[:, 1], team, pit_xy, d_pit, alive)
    arr = np.sort(eta)
    reach = np.where(alive[:, None], nav.d(nodes[:, None], pit_cells[None, :]) / speed[:, None], np.inf).min(0)
    per_role = {}
    for r, (px, py), a, dp, n, z in zip(io.ROLE_SHORT, xy, alive, d_pit, nodes, zones):
        per_role.update({f"x_{r}": px, f"y_{r}": py, f"alive_{r}": bool(a), f"dpit_{r}": dp,
                         f"node_{r}": int(n), f"zone_{r}": z})
    return {
        **per_role,
        "alive": int(alive.sum()),
        "partition": label(blocks),
        "block_zones": "|".join(geometry.block_zones(blocks, D, d_pit, zones)),
        "shape": "+".join(str(len(b)) for b in blocks),
        "n_blocks": len(blocks),
        "merge_heights": ",".join(f"{h:.0f}" for h in merge_heights(D, alive)),
        "k_pit": int(((d_pit <= geometry.PIT_R) & alive).sum()),
        **{f"tau{k}": float(arr[k - 1]) for k in (1, 3, 5)},
        "_reach": reach,
    }


def main():
    nav = NavGraph()
    sp = io.spawns()
    pits = geometry.pit_centres(sp)
    rows, excluded = [], []
    for gid, gsp in sp.groupby("RiotPlatformGameId"):
        frames = io.load_v5(gid)["frames"]
        events = [e for f in frames for e in f["events"]]
        reveal = next(((e["name"], e["timestamp"]) for e in events
                       if e["type"] == "DRAGON_SOUL_GIVEN" and e.get("teamId") == 0), (None, None))
        deaths = {}
        for e in events:
            if e["type"] == "CHAMPION_KILL":
                deaths.setdefault(e["victimId"], []).append((e["timestamp"], e["position"]["x"], e["position"]["y"]))
        for _, s in gsp.iterrows():
            if s["objective"] not in FIXED or s["idx"] != 1:
                continue  # R layer = fixed-time first spawns only (exact frames)
            pit = geometry.pit_of(s["objective"], s["kill_x"], s["kill_y"])
            if pit is None:
                raise ValueError(f"untaken Atakhan in {gid}: pit side needs a pre-window rule")
            ks = [int(round(s["spawn_s"] / 60)) + off // 60 for off in OFFSETS]
            if any(k >= len(frames) - 1 or abs(frames[k]["timestamp"] - 1000 * (s["spawn_s"] + off)) > GRID_TOL_MS
                   for k, off in zip(ks, OFFSETS)):
                excluded.append({"RiotPlatformGameId": gid, "objective": s["objective"], "reason": "off-grid or game-end frame"})
                continue
            pit_xy = pits.loc[pit, ["x", "y"]].to_numpy(float)
            pit_node = nav.node(*pit_xy)[0]
            pit_cells = np.flatnonzero(nav.dist[pit_node] <= geometry.PIT_R)
            elem_pre = reveal[0] if reveal[1] is not None and reveal[1] < 1000 * (s["spawn_s"] - 60) else None
            for k, off in zip(ks, OFFSETS):
                fr = frames[k]
                gold = {t: sum(fr["participantFrames"][str(i)]["totalGold"] for i in io.TEAM_IDS[t]) for t in (100, 200)}
                snap = {t: team_snapshot(nav, fr, t, pit_xy, pit_node, pit_cells, deaths) for t in (100, 200)}
                for t in (100, 200):
                    mine, theirs = snap[t]["_reach"], snap[300 - t]["_reach"]
                    grub_share = (s[f"grubs_{t}"] / 3) if s["objective"] == "HORDE" and pd.notna(s.get(f"grubs_{t}")) else np.nan
                    rows.append({
                        "RiotPlatformGameId": gid, "GameId": s["GameId"], "event": s["event"],
                        "objective": s["objective"], "pit": pit, "spawn_s": s["spawn_s"],
                        "offset_s": off, "frame_ms": fr["timestamp"], "side": t,
                        "team": s["Blue"] if t == 100 else s["Red"], "opponent": s["Red"] if t == 100 else s["Blue"],
                        "taken_by_team": (s["taken_by"] == t) if pd.notna(s["taken_by"]) else np.nan,
                        "grub_share": grub_share, "kill_s": s["kill_s"],
                        "obj_up": bool(pd.isna(s["kill_s"]) or 1000 * s["kill_s"] > fr["timestamp"]),
                        "gold_diff": gold[t] - gold[300 - t],
                        "rift_element_pre": elem_pre, "rift_reveal_s": reveal[1] / 1000 if reveal[1] else np.nan,
                        "rift_transformed": bool(reveal[1] is not None and reveal[1] <= fr["timestamp"]),
                        "pit_control": float(np.mean(mine < theirs) + 0.5 * np.mean(mine == theirs)),
                        **{k2: v for k2, v in snap[t].items() if k2 != "_reach"},
                        **{f"opp_{k2}": snap[300 - t][k2] for k2 in ("alive", "k_pit", "tau1", "tau3")},
                    })
    df = pd.DataFrame(rows)
    out = io.ROOT / "fits" / "r_layer_setups.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (io.ROOT / "fits" / "r_layer_excluded.json").write_text(json.dumps(excluded, indent=1))
    n_setups = df.groupby(["RiotPlatformGameId", "objective", "side"]).ngroups
    print(f"{len(df)} team-snapshots ({n_setups} team-setups); excluded spawns: {len(excluded)}; "
          f"phantom-dead corrected: {int(df['phantom_dead'].sum())} player-snapshots -> {out}")


if __name__ == "__main__":
    main()
