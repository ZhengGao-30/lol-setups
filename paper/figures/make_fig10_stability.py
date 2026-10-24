"""Fig. 10: HASSE vs the TAGNet port. (a) agreement of repeated fits (HASSE: cross-restart ARI of full-data
solutions; TAGNet: cross-seed ARI), (b) series-bootstrap ARI distributions (HASSE refits the whole model;
TAGNet re-runs only its K-Means step), (c) leakage control: Cramer's V of the assignment with the gold tercile."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    s5, s6 = js("results/hasse_v/stability_K5.json"), js("results/hasse_v/stability_K6.json")
    tg = js("results/baseline/tagnet_summary.json")
    b5, b6 = np.load(src("results/hasse_v/bootstrap_ari_K5.npy")), np.load(src("results/hasse_v/bootstrap_ari_K6.npy"))
    tn, tc = np.load(src("results/baseline/tagnet_noctx_bootstrap_ari.npy")), np.load(src("results/baseline/tagnet_ctx_bootstrap_ari.npy"))
    lk = csv("results/context/leakage_control_K5.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.5), gridspec_kw=dict(wspace=0.75, width_ratios=[1.3, 1.5, 0.9]))
    a = axs[0]
    vals = [s5["cross_restart_ari_mean"], s6["cross_restart_ari_mean"], tg["noctx_pairwise_ari"], tg["ctx_pairwise_ari"]]
    labs = ["HASSE\nK=5", "HASSE\nK=6", "TAGNet", "TAGNet\n+ctx"]
    a.bar(range(4), vals, color=[C["blue"], C["blue"], C["grey"], C["grey"]])
    a.set_xticks(range(4), labs, fontsize=5.8)
    a.set_ylim(0, 1)
    a.set_ylabel("ARI between fits")
    a.set_title("(a) repeat fits || ARI between runs")
    b = axs[1]
    parts = b.violinplot([b5, b6, tn, tc], showmedians=True, widths=0.8)
    for pc, col in zip(parts["bodies"], [C["blue"], C["blue"], C["grey"], C["grey"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
    b.set_xticks(range(1, 5), [f"HASSE\nK=5\nn={len(b5)}", f"HASSE\nK=6\nn={len(b6)}", "TAGNet\nK-Means", "+ctx\nK-Means"], fontsize=5.6)
    b.set_ylabel("series-bootstrap ARI")
    b.set_title("(b) series bootstrap || ARI to the full fit")
    c = axs[2]
    c.barh(range(len(lk)), lk.cramers_V_gold, color=[C["blue"], C["sky"], C["grey"], C["red"]][:len(lk)])
    c.set_yticks(range(len(lk)), [{"HASSE cut": "HASSE (cut)", "TAGNet without context": "TAGNet"}.get(m, "HASSE leaky" if "leaky" in m else ("TAGNet + context" if "TAGNet" in m else m)) for m in lk.model], fontsize=6.0)
    c.invert_yaxis()
    c.set_xlabel("Cramér's V\n(gold tercile)")
    c.set_title("(c) leakage || context vs patterns")
    N["hsCrossKFive"] = f"{s5['cross_restart_ari_mean']:.3f}"
    N["hsCrossKSix"] = f"{s6['cross_restart_ari_mean']:.3f}"
    N["hsBootKFive"] = f"{s5['bootstrap_ari_mean']:.3f}"
    N["hsBootKFiveQ"] = f"{s5['bootstrap_ari_q05']:.3f}"
    N["hsBootKFiveN"] = f"{s5['bootstrap_n']}"
    N["hsBootKSix"] = f"{s6['bootstrap_ari_mean']:.3f}"
    N["hsBootKSixQ"] = f"{s6['bootstrap_ari_q05']:.3f}"
    N["hsBootKSixN"] = f"{s6['bootstrap_n']}"
    N["tgN"] = f"{tg['n_setups']:,}"
    N["tgSeedAri"] = f"{tg['noctx_pairwise_ari']:.3f}"
    N["tgSeedAriCtx"] = f"{tg['ctx_pairwise_ari']:.3f}"
    N["tgBootAri"] = f"{tg['noctx_bootstrap_ari_mean']:.3f}"
    N["tgBootAriQ"] = f"{tg['noctx_bootstrap_ari_q05']:.3f}"
    N["tgBootAriCtx"] = f"{tg['ctx_bootstrap_ari_mean']:.3f}"
    N["tgBootAriCtxQ"] = f"{tg['ctx_bootstrap_ari_q05']:.3f}"
    N["tgCountAri"] = f"{tg['count_baseline_vs_tagnet_ari']:.3f}"
    N["tgVgold"] = f"{tg['noctx_V_gold_mean']:.3f}"
    N["tgVgoldCtx"] = f"{tg['ctx_V_gold_mean']:.3f}"
    N["tgLeakRatio"] = f"{tg['leakage_inflation_V_gold']:.2f}"
    lk_ = lk.set_index("model")
    N["lkHasseCut"] = f"{lk_.cramers_V_gold.iloc[0]:.3f}"
    if len(lk) > 3:
        N["lkHasseLeaky"] = f"{lk_.cramers_V_gold.iloc[1]:.3f}"
    return save(fig, "fig10_stability")


if __name__ == "__main__":
    run_main(build)
