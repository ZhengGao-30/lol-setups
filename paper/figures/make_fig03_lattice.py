"""Fig. 3: the partition lattice Pi_5 as a Hasse diagram (52 partitions, 160 covering edges), nodes sized by
their share of V-layer team-snapshots at the spawn; the lattice counts are computed, not typed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403
from collections import Counter  # noqa: E402

ROLES = "TJMBS"


def to_str(p):
    return "|".join("".join(ROLES[i] for i in b) for b in p)


def build(N):
    L = lattice_mod()
    P = L.partitions((0, 1, 2, 3, 4))
    edges = [(a, b) for a in P for b in P if a != b and len(b) == len(a) - 1 and L.join(a, b) == b]
    D = np.array([[L.dist(a, b) for b in P] for a in P])
    shapes = Counter("+".join(str(len(b)) for b in p) for p in P)
    N["latticeN"] = f"{len(P)}"
    N["latticeEdges"] = f"{len(edges)}"
    N["latticeDiam"] = f"{int(D.max())}"
    N["latticeShapes"] = f"{len(shapes)}"

    df = pd.read_parquet(src("fits/v_layer_setups.parquet"), columns=["offset_s", "alive", "partition"])
    at = df[(df.offset_s == 0) & (df.alive == 5)]
    share = at.partition.value_counts(normalize=True)
    N["latticeObservedAtSpawn"] = f"{int(share.index.isin([to_str(p) for p in P]).sum())}"
    allshare = df[df.alive == 5].partition.value_counts(normalize=True)
    N["latticeObservedAll"] = f"{int(allshare.index.isin([to_str(p) for p in P]).sum())}"

    cov = pd.read_parquet(src("fits/v_layer_setups.parquet"), columns=["RiotPlatformGameId", "objective", "idx", "side", "offset_s"])
    N["vSpawnsCovered"] = f"{cov.groupby(['RiotPlatformGameId', 'objective', 'idx']).ngroups:,}"
    per = cov.groupby(["RiotPlatformGameId", "objective", "idx", "side"]).offset_s.nunique()
    N["vSnapMedPct"] = f"{100 * per.median() / cov.offset_s.nunique():.0f}"
    N["vGridSnaps"] = f"{cov.offset_s.nunique()}"
    levels = {}
    for p in P:
        levels.setdefault(len(p), []).append(p)
    pos = {}
    for nb, ps in levels.items():
        ps = sorted(ps, key=lambda p: (sorted([len(b) for b in p], reverse=True), to_str(p)))
        for i, p in enumerate(ps):
            pos[p] = ((i - (len(ps) - 1) / 2) * (1.0 if len(ps) < 12 else 0.62), 5 - nb)
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.45))
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=C["grey"], lw=0.3, alpha=0.55, zorder=1)
    s = np.array([share.get(to_str(p), 0.0) for p in P])
    xs = np.array([pos[p][0] for p in P])
    ys = np.array([pos[p][1] for p in P])
    ax.scatter(xs, ys, s=6 + 1500 * s, c=C["blue"], alpha=0.85, edgecolors="white", linewidths=0.3, zorder=2)
    seen = {}
    for p in sorted(P, key=lambda q: pos[q][0]):
        if share.get(to_str(p), 0) > 0.03 or len(p) in (1, 5):
            k = seen.get(len(p), 0)
            seen[len(p)] = k + 1
            side = 1 if pos[p][0] >= 0 else -1
            dy = 0.2 if k % 2 == 0 else -0.34
            ax.text(pos[p][0] + side * 0.18, pos[p][1] + dy, to_str(p), ha="left" if side > 0 else "right", fontsize=6.6, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
    ax.set_yticks(range(5), [f"{5 - i} block{'s' if 5 - i > 1 else ''}" for i in range(5)])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(-0.4, 4.5)
    return save(fig, "fig03_lattice")


if __name__ == "__main__":
    run_main(build)
