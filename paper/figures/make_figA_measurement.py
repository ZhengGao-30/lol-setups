"""Appendix figures on measurement and the R layer: per-game tracker quality, formation shapes at the spawn,
the 4+1 singleton role, and the full-vs-clean sensitivity of the V-layer contrasts."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def pct(s):
    m = re.match(r"\s*(-?[\d.]+)%\s*\[(-?[\d.]+)%,\s*(-?[\d.]+)%\]", s)
    return [float(x) / 100 for x in m.groups()]


def build(N):
    out = []
    pg = csv("results/tracker/summary_per_game.csv")
    fpg = csv("results/tracker/summary_formation_per_game.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.0), gridspec_kw=dict(wspace=0.5))
    axs[0].hist(pg.q50, bins=np.logspace(np.log10(50), np.log10(3000), 30), color=C["blue"])
    axs[0].set_xscale("log")
    axs[0].set_xlabel("per-game median error")
    axs[0].set_ylabel("games")
    axs[1].hist(pg.lt600, bins=20, color=C["sky"])
    axs[1].axvline(0.6, color="k", lw=0.5, ls=":")
    axs[1].set_xlabel("per-game share error $<$ 600")
    axs[2].hist(fpg.exact, bins=20, color=C["orange"], alpha=0.8, label="exact")
    axs[2].hist(fpg.rand, bins=20, color=C["purple"], alpha=0.6, label="Rand")
    axs[2].set_xlabel("per-game formation agreement")
    axs[2].legend(fontsize=6.3)
    out.append(save(fig, "figA_tracker_pergame"))

    sh = csv("results/r_layer/shape_shares.csv")
    shapes = ["5", "4+1", "3+2", "3+1+1", "2+2+1", "2+1+1+1", "1+1+1+1+1"]
    objs = ["DRAGON", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR"]
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 2.4), gridspec_kw=dict(wspace=0.55, width_ratios=[2, 1]))
    a = axs[0]
    offs = [-60, 0, 60]
    xt, xl = [], []
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("sh", ["#2F5D96", "#8FAED6", "#DCE6F2", "#F0E2BD", "#E3C48A", "#C98A1B", "#9A5A1A"], N=len(shapes))
    for i, o in enumerate(objs):
        for j, off in enumerate(offs):
            x = i * 3.6 + j
            bottom = 0
            d = sh[(sh.objective == o) & (sh.offset_s == off)].set_index("shape").share
            for k, s in enumerate(shapes):
                v = d.get(s, 0.0)
                a.bar(x, v, bottom=bottom, color=cm(k), width=0.85, label=s if (i == 0 and j == 0) else None)
                bottom += v
            xt.append(x)
            xl.append({-60: "−60", 0: "s", 60: "+60"}[off])
        a.text(i * 3.6 + 1, 1.04, OBJ_LABEL.get(o, o).replace("First ", ""), ha="center", fontsize=6.5)
    a.set_xticks(xt, xl, fontsize=6.0)
    a.set_ylabel("share of team-snapshots")
    a.legend(fontsize=6.0, ncol=1, bbox_to_anchor=(1.0, 1.0), loc="upper left", title="shape", title_fontsize=6.3)
    a.set_ylim(0, 1.1)
    b = axs[1]
    sr = csv("results/r_layer/four_plus_one_singleton_role.csv")
    vals = np.array([pct(s) for s in sr.share])
    b.bar(sr.role, vals[:, 0], color=C["blue"], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], capsize=2)
    b.axhline(0.2, color="k", lw=0.6, ls=":")
    b.set_ylabel("share of 4+1 singletons")
    b.set_title("lone player in 4+1 || R layer, uniform = 20%")
    out.append(save(fig, "figA_shapes_singleton"))
    s_ = sr.set_index("role")
    for r_, k in (("T", "Top"), ("M", "Mid"), ("B", "Bot"), ("J", "Jgl"), ("S", "Sup")):
        v = pct(s_.loc[r_, "share"])
        N[f"single{k}"] = f(v[0], 1, pct=True)
        N[f"single{k}CI"] = ci(v[1], v[2], 1, pct=True)

    con = csv("results/v_layer/contrasts_trailing_minus_leading.csv")
    cl = csv("results/v_layer/contrasts_trailing_minus_leading_clean.csv")
    m = con.merge(cl, on=["group", "offset_s", "R", "stat"], suffixes=("", "_c"))
    m = m[m.R == 1500]
    fig, a = plt.subplots(figsize=(TEXT_W * 0.6, 2.6))
    a.scatter(m.trailing_minus_leading, m.trailing_minus_leading_c, s=3, c=np.where(m.headline, C["red"], C["grey"]))
    lim = [m[["trailing_minus_leading", "trailing_minus_leading_c"]].min().min(), m[["trailing_minus_leading", "trailing_minus_leading_c"]].max().max()]
    a.plot(lim, lim, color="k", lw=0.5)
    a.set_xlabel("contrast, all snapshots")
    a.set_ylabel("contrast, mostly-detected only")
    out.append(save(fig, "figA_clean_vs_full"))
    N["cleanCorr"] = f"{np.corrcoef(m.trailing_minus_leading.fillna(0), m.trailing_minus_leading_c.fillna(0))[0, 1]:.2f}"
    return out


if __name__ == "__main__":
    run_main(build)
