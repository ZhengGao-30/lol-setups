"""Fig. 1, "The paper in one picture": one real setup (Worlds 2025 final, game 4, first baron) with every object of
the paper in three panels: the walled map with the geodesic Voronoi front and the formations (the map),
the walks on the partition lattice and the allocation to the pit (the setup), and the k-arrival times with the
LB-QRE decision (the choice).

The walkable-cell grid is rebuilt from data/navgraph/nav_nodes.csv (the 12,944 walkable 100-unit cells that
the original fits/navgraph_base.h5 was built from; that file is not needed any more): geodesic distances are
Dijkstra distances on the 16-neighbourhood graph of those cells, exactly as in the measurement layer. All player
positions, partitions, k, arrival times, W and the LB-QRE actions are read from the fitted result files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402

GAME = "2025 Season World Championship/Main Event_Finals_1_4"
OBJ, IDX, OFF = "BARON_NASHOR", 1, 60
ROLES = "TJMBS"
SPEED = 381.0  # median movement speed (units/s), as in lolsetup.qre_data
PIT = (5007.0, 10471.0)  # centre of the baron pit (map units)
BLUE, AMBER = "#2F5D96", "#B7791F"
MOVES = [(1, 0), (0, 1), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]


def blocks(part):
    return [b for b in str(part).split("|") if b]


def geodesic_owner(nodes_by_side):
    """Walkable grid from nav_nodes.csv and, for each cell, which side reaches it first (geodesic distance)."""
    nn = csv("navgraph/nav_nodes.csv")
    ci, cj = nn.cell_i.to_numpy(), nn.cell_j.to_numpy()
    idx = -np.ones((ci.max() + 3, cj.max() + 3), int)
    idx[ci, cj] = nn.node.to_numpy()
    rows, cols, w = [], [], []
    for dx, dy in MOVES:
        a = idx[ci, cj]
        i2, j2 = ci + dx, cj + dy
        ok = (i2 >= 0) & (j2 >= 0) & (i2 < idx.shape[0]) & (j2 < idx.shape[1])
        b = np.where(ok, idx[np.clip(i2, 0, idx.shape[0] - 1), np.clip(j2, 0, idx.shape[1] - 1)], -1)
        m = ok & (b >= 0)
        rows.append(a[m])
        cols.append(b[m])
        w.append(np.full(m.sum(), 100.0 * np.hypot(dx, dy)))
    n = len(nn)
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    d = {s: dijkstra(G, directed=False, indices=ns, min_only=True) for s, ns in nodes_by_side.items()}
    own = np.full(idx.shape, -1)
    own[ci, cj] = np.where(d[100] < d[200], 0, 1)
    walk = idx >= 0
    return own, walk


def build(N):
    df = pd.read_parquet(src("fits/v_layer_setups.parquet"))
    g = df[(df.GameId == GAME) & (df.objective == OBJ) & (df.idx == IDX)]
    if g.empty:
        raise Missing(f"{GAME} {OBJ} {IDX}")
    snap = g[g.offset_s == OFF].sort_values("side")
    nodes = {r.side: [int(r[f"node_{x}"]) for x in ROLES if r[f"alive_{x}"]] for _, r in snap.iterrows()}
    own, walk = geodesic_owner(nodes)
    teams = {r.side: r.team for _, r in snap.iterrows()}
    col = {100: BLUE, 200: AMBER}
    tint = {0: np.array(matplotlib.colors.to_rgb("#B9CBE6")), 1: np.array(matplotlib.colors.to_rgb("#EBCF9B"))}

    fig = plt.figure(figsize=(TEXT_W, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1.0, 1.05], wspace=0.42, left=0.0, right=1.0, top=0.80, bottom=0.12)

    # ---------- panel 1: the map ----------
    ax = fig.add_subplot(gs[0, 0])
    img = np.ones(own.shape + (3,)) * 0.93  # walls
    for o, c in tint.items():
        img[own == o] = c
    edge = np.zeros(own.shape, bool)
    for dx, dy in ((1, 0), (0, 1)):
        a, b = own[:-dx or None, :-dy or None], own[dx:, dy:]
        e = (a >= 0) & (b >= 0) & (a != b)
        edge[:-dx or None, :-dy or None] |= e
    img[edge] = np.array(matplotlib.colors.to_rgb("#333333"))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[0, 100 * own.shape[0], 0, 100 * own.shape[1]],
              interpolation="nearest")
    for _, r in snap.iterrows():
        c = col[r.side]
        P = {x: (r[f"x_{x}"], r[f"y_{x}"]) for x in ROLES if r[f"alive_{x}"]}
        for b in blocks(r.partition):
            pts = [P[x] for x in b if x in P]
            for i in range(1, len(pts)):
                ax.plot([pts[0][0], pts[i][0]], [pts[0][1], pts[i][1]], color=c, lw=1.0, alpha=0.9, zorder=2)
        for x, (px, py) in P.items():
            ax.scatter(px, py, s=40, c="white", edgecolors=c, linewidths=1.2, zorder=3)
            ax.text(px, py, x, ha="center", va="center", fontsize=5.6, color=c, zorder=4, fontweight="bold")
    ax.scatter(*PIT, marker="*", s=70, c="#3a3a3a", zorder=5, edgecolors="white", linewidths=0.4)
    ax.text(PIT[0] - 450, PIT[1] + 550, "pit", ha="right", fontsize=6.5, style="italic", color="#333333", zorder=5)
    ax.plot([], [], color="#333333", lw=1.0, label="geodesic Voronoi front")
    ax.set_xlim(0, 15000)
    ax.set_ylim(0, 15000)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#bbbbbb")
    ax.set_title("the map || who gets there first, and how they stand")
    ax.text(0.5, -0.045, f"{teams[100]} (blue) vs {teams[200]} (amber), baron $s{OFF:+d}$ s", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.2, color="#444444")
    ax.legend(loc="lower right", fontsize=5.6, handlelength=1.2, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.85)

    # ---------- panel 2: the walk on the lattice and the allocation ----------
    sub = gs[0, 1].subgridspec(2, 1, hspace=0.22)
    axc = fig.add_subplot(sub[0])
    axd = fig.add_subplot(sub[1], sharex=axc)
    for side, h in g.groupby("side"):
        h = h.sort_values("offset_s")
        nb = h.partition.map(lambda p: len(blocks(p)) if p else np.nan)
        axc.plot(h.offset_s, nb, "-o", ms=2.2, lw=1.1, color=col[side], label=teams[side])
        axd.plot(h.offset_s, h.k_pit, "-o", ms=2.2, lw=1.1, color=col[side])
    axc.set_ylim(5.4, 0.6)
    axc.set_yticks([1, 2, 3, 4, 5])
    axc.set_ylabel(r"blocks $|\pi_t|$")
    axc.tick_params(labelbottom=False)
    axd.set_ylabel("players at pit $k$")
    axd.set_xlabel("time from spawn (s)")
    for a in (axc, axd):
        a.axvline(0, color="#999999", lw=0.6, ls=":")
    axc.set_title(r"the setup || walks on $\Pi_5$ (up = merge), then the pit")

    # ---------- panel 3: the choice ----------
    sub3 = gs[0, 2].subgridspec(2, 1, hspace=0.9, height_ratios=[1.0, 1.25])
    axb = fig.add_subplot(sub3[0])
    for _, r in snap.iterrows():
        d = np.sort([r[f"dpit_{x}"] for x in ROLES if r[f"alive_{x}"]]) / SPEED
        axb.step(np.arange(1, len(d) + 1), d, where="mid", color=col[r.side], lw=1.1)
        axb.plot(np.arange(1, len(d) + 1), d, "o", ms=2.6, color=col[r.side])
    axb.set_xlabel("$k$ (players)")
    axb.set_ylabel(r"$\tau^{(k)}$ to pit (s)")
    axb.set_xticks([1, 2, 3, 4, 5])
    axb.set_title(r"the choice || $k$-arrival times, then LB-QRE")

    axe = fig.add_subplot(sub3[1])
    axe.axis("off")
    led = csv("results/t1/A7_worlds2025_final_ledger.csv")
    gid = g.RiotPlatformGameId.iloc[0]
    row = led[(led.game == gid) & (led.objective == OBJ) & (led.idx == IDX)]
    if len(row):
        r = row.iloc[0]
        axe.add_patch(matplotlib.patches.FancyBboxPatch((0.0, 0.02), 1.0, 0.96, boxstyle="round,pad=0.0,rounding_size=0.06",
                                                        transform=axe.transAxes, fc="#F6F6F6", ec="#C8C8C8", lw=0.6))
        txt = (f"action $(k,m,\\tau)$ at $t_c$\n"
               f"{teams[100]}: ({int(r.blue_k)}, {int(r.blue_m)}, {int(r.blue_tau)})\n"
               f"{teams[200]}: ({int(r.red_k)}, {int(r.red_m)}, {int(r.red_tau)})\n"
               f"$W_{{blue}}$ = {r.W_blue:.2f},  $\\hat p_{{blue}}$ = {r.p_hat_blue:.2f}\n"
               f"value to blue = {100 * r.value_obj_blue:+.1f} pts")
        axe.text(0.06, 0.9, txt, va="top", fontsize=6.1, family="serif", transform=axe.transAxes, linespacing=1.3)
        N["teaserWblue"] = f"{r.W_blue:.2f}"
    N["teaserGame"] = "game 4 of the Worlds 2025 final"
    N["teaserBlue"], N["teaserRed"] = teams[100], teams[200]
    return save(fig, "fig01_teaser", fill=False)


if __name__ == "__main__":
    run_main(build)
