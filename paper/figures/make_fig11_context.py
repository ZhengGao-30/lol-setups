"""Fig. 11: context effects on pattern occupancy (cut design: context enters only after the patterns are fixed).
Average marginal effect per SD of each covariate on each pattern's occupancy, 500 series-block resamples."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

FEAT = {"gold_k": "gold difference", "blue": "blue side", "minute": "game minute", "own_soul_pt": "own soul point",
        "opp_soul_pt": "opp. soul point", "elo_diff": "Elo difference", "cls1": "class: Atakhan", "cls2": "class: baron/elder"}


def build(N, K=6, name="fig11_context"):
    d = csv(f"results/context/ame_v_K{K}.csv")
    feats = [x for x in FEAT if x in set(d.feature)]
    fig, axs = plt.subplots(1, len(feats), figsize=(TEXT_W, 2.3), sharey=True, gridspec_kw=dict(wspace=0.14))
    for a, ft in zip(axs, feats):
        g = d[d.feature == ft].sort_values("pattern")
        y = g.pattern.to_numpy()
        cols = [C["blue"] if s else C["grey"] for s in g.sig]
        a.errorbar(g.ame_per_sd, y, xerr=[g.ame_per_sd - g.lo, g.hi - g.ame_per_sd], fmt="none", ecolor=cols, lw=0.8)
        a.scatter(g.ame_per_sd, y, s=7, c=cols, zorder=3)
        a.axvline(0, color="k", lw=0.5)
        a.set_title(FEAT[ft].replace(": ", ":\n") if ":" in FEAT[ft] else FEAT[ft].replace(" ", "\n", 1))
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        a.tick_params(axis="x", labelsize=6.0)
    axs[0].set_yticks(range(K), [f"P{i}" for i in range(K)])
    axs[0].invert_yaxis()
    fig.supxlabel("average marginal effect on occupancy per SD", fontsize=7.2, y=-0.06)
    if K == 6:
        r = d.set_index(["pattern", "feature"])
        for p, nm in ((5, "Five"), (0, "Zero"), (4, "Four")):
            N[f"ameGold{nm}"] = f(r.loc[(p, "gold_k"), "ame_per_sd"], 3, sign=True)
            N[f"ameGold{nm}CI"] = ci(r.loc[(p, "gold_k"), "lo"], r.loc[(p, "gold_k"), "hi"], 3)
        N["ameSoulFour"] = f(r.loc[(4, "own_soul_pt"), "ame_per_sd"], 3, sign=True)
        N["ameSoulFourCI"] = ci(r.loc[(4, "own_soul_pt"), "lo"], r.loc[(4, "own_soul_pt"), "hi"], 3)
        N["ameSigCount"] = f"{int(d.sig.sum())}"
        N["ameCells"] = f"{len(d)}"
    return save(fig, name)


if __name__ == "__main__":
    run_main(build)
