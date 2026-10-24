"""Walkable navigation graph of Summoner's Rift (one map version, base terrain) and geodesic distances.

Map version: Season 14+ Rift (patches 14.1 onward; 2024-2026 games), which the core set (2025) uses.
Terrain: base terrain only -- frames after the rift transformation (the teamId-0 DRAGON_SOUL_GIVEN
event) are excluded, because elemental terrain changes walls. Terrain-specific tables come later.

Walkable mask: a CELL-unit grid cell is walkable if participant-frame positions fall in it in at
least K_SUPPORT distinct games. Event positions (kills during dashes/wall hops, turret footprints)
are excluded, and multi-game support removes one-off in-wall samples that would create geodesic
shortcuts. Rarely visited corners may be missing; we report the mask's sensitivity (k = 2, 3, 5) as
the change in geodesic distance from each pit centre.

Graph: 16-neighbourhood (orthogonal, diagonal, knight moves) with Euclidean step lengths; diagonal
and knight steps only if every cell they cut through is walkable (no corner cutting). An
8-neighbourhood would overestimate straight-line paths by up to 1/cos(22.5 deg) ~ 8.2%.

Usage:  python -m lolsetup.navgraph build   -> fits/navgraph_base.h5
"""
import sys

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from . import geometry, io

CELL = 100
SIZE = 15000
N = SIZE // CELL  # 150 x 150 grid
STEPS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2)]  # half of the 16 (undirected)
K_SUPPORT = 3
MAP_VERSION = "S14+ Rift (2024-2026), base terrain"


def cell_of(x, y):
    return np.clip((np.asarray(x) // CELL).astype(int), 0, N - 1), np.clip((np.asarray(y) // CELL).astype(int), 0, N - 1)


def support(games):
    """(raw position counts, number of distinct games per cell), base-terrain participant frames only."""
    occ = np.zeros((N, N), dtype=np.int64)
    seen = np.zeros((N, N), dtype=np.int64)
    for gid in games["RiotPlatformGameId"]:
        frames = io.load_v5(gid)["frames"]
        transform = next((e["timestamp"] for f in frames for e in f["events"]
                          if e["type"] == "DRAGON_SOUL_GIVEN" and e.get("teamId") == 0), np.inf)
        xs, ys = [], []
        for f in frames:
            if f["timestamp"] >= transform:
                break
            for p in f["participantFrames"].values():
                xs.append(p["position"]["x"]); ys.append(p["position"]["y"])
        if not xs:
            continue
        cx, cy = cell_of(xs, ys)
        np.add.at(occ, (cx, cy), 1)
        g = np.zeros((N, N), dtype=bool)
        g[cx, cy] = True
        seen += g
    return occ, seen


def _cut_cells(i, j, di, dj):
    """Cells a step from (i, j) by (di, dj) passes through (excluding endpoints)."""
    if abs(di) + abs(dj) == 1:
        return []
    if abs(di) == 1 and abs(dj) == 1:
        return [(i + di, j), (i, j + dj)]
    if abs(di) == 2:  # knight step along x
        s = di // 2
        return [(i + s, j), (i + s, j + dj)]
    s = dj // 2       # knight step along y
    return [(i, j + s), (i + di, j + s)]


def build_graph(mask):
    idx = -np.ones((N, N), dtype=np.int64)
    cells = np.argwhere(mask)
    idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    rows, cols, w = [], [], []
    for i, j in cells:
        for di, dj in STEPS:
            a, b = i + di, j + dj
            if not (0 <= a < N and 0 <= b < N) or not mask[a, b]:
                continue
            if any(not (0 <= c < N and 0 <= d < N and mask[c, d]) for c, d in _cut_cells(i, j, di, dj)):
                continue
            rows.append(idx[i, j]); cols.append(idx[a, b]); w.append(CELL * np.hypot(di, dj))
    g = csr_matrix((w, (rows, cols)), shape=(len(cells), len(cells)))
    return g, idx, cells


def giant(mask):
    g, idx, cells = build_graph(mask)
    ncomp, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels)
    keep = labels == np.argmax(sizes)
    m2 = np.zeros_like(mask)
    m2[cells[keep, 0], cells[keep, 1]] = True
    return m2, ncomp, float(sizes.max() / sizes.sum())


def pit_distance_fields(mask, pits):
    g, idx, cells = build_graph(mask)
    out = {}
    for name, (x, y) in pits.iterrows():
        cx, cy = cell_of(x, y)
        src = idx[cx, cy] if idx[cx, cy] >= 0 else int(np.argmin((cells[:, 0] - cx) ** 2 + (cells[:, 1] - cy) ** 2))
        field = np.full((N, N), np.nan)
        field[cells[:, 0], cells[:, 1]] = dijkstra(g, indices=src, directed=False)
        out[name] = field
    return out


def build(out=None):
    games = io.archive_games()
    games = games[games["Year"] >= 2024]
    occ, seen = support(games)
    pits = geometry.pit_centres(io.spawns())
    masks = {k: giant(seen >= k) for k in (2, K_SUPPORT, 5)}
    mask, ncomp, share = masks[K_SUPPORT]
    # sensitivity of pit geodesics to the support threshold (vs the primary k)
    ref = pit_distance_fields(mask, pits)
    sens = {}
    for k in (2, 5):
        alt = pit_distance_fields(masks[k][0], pits)
        for p in ref:
            both = np.isfinite(ref[p]) & np.isfinite(alt[p])
            dlt = np.abs(alt[p][both] - ref[p][both])
            sens[f"k{k}_{p}_median_abs_diff"] = float(np.median(dlt))
            sens[f"k{k}_{p}_p99_abs_diff"] = float(np.percentile(dlt, 99))
    g, idx, cells = build_graph(mask)
    dist = dijkstra(g, directed=False)
    out = out or (io.ROOT / "fits" / "navgraph_base.h5")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        h["occupancy"] = occ
        h["games_seen"] = seen
        h["mask"] = mask
        h["node_index"] = idx
        h["cells"] = cells
        h.create_dataset("dist", data=np.minimum(dist, 65535).astype(np.uint16), compression="gzip")
        h.attrs.update(cell=CELL, size=SIZE, games=len(games), k_support=K_SUPPORT, map_version=MAP_VERSION,
                       years="2024,2025,2026", components_before=ncomp, giant_share=share, nodes=len(cells),
                       **{f"cells_k{k}": int(m[0].sum()) for k, m in masks.items()}, **sens)
    print(f"games {len(games)}; positions {int(occ.sum())}; cells k2/k3/k5 = "
          f"{[int(m[0].sum()) for m in masks.values()]}; giant share {share:.3%}; -> {out}")
    print({k: round(v, 1) for k, v in sens.items()})
    return out


class NavGraph:
    """Geodesic distances between map points via the precomputed all-pairs table."""

    def __init__(self, path=None):
        path = path or (io.ROOT / "fits" / "navgraph_base.h5")
        with h5py.File(path, "r") as h:
            if h.attrs.get("map_version") != MAP_VERSION:
                raise ValueError(f"navgraph map_version {h.attrs.get('map_version')!r} != {MAP_VERSION!r}")
            self.mask = h["mask"][:]
            self.idx = h["node_index"][:]
            self.cells = h["cells"][:]
            self.dist = h["dist"][:]

    def node(self, x, y):
        """Nearest walkable node to each (x, y) (handles points snapped into walls)."""
        cx, cy = cell_of(x, y)
        cx, cy = np.atleast_1d(cx), np.atleast_1d(cy)
        out = self.idx[cx, cy]
        bad = out < 0
        if bad.any():
            d = (self.cells[None, :, 0] - cx[bad, None]) ** 2 + (self.cells[None, :, 1] - cy[bad, None]) ** 2
            out[bad] = np.argmin(d, axis=1)
        return out

    def d(self, a, b):
        """Geodesic distance (map units) between node arrays a and b (broadcast)."""
        return self.dist[np.asarray(a), np.asarray(b)].astype(float)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
