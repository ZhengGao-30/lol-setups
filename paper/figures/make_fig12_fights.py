"""Fig. 12: fight laws. (a) S7: probability that the next death falls on the larger side, by local numbers,
against the stochastic square law; (b) Tullock fight-law parameters under the two count definitions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    s7 = csv("results/fights/s7_binned.csv")
    fl = csv("results/fights/fight_laws.csv")
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 1.95), gridspec_kw=dict(wspace=0.4, width_ratios=[2.6, 1]))
    a = axs[0]
    x = np.arange(len(s7))
    a.bar(x - 0.2, s7.p_victim_larger, 0.4, color=C["blue"], label="observed")
    a.bar(x + 0.2, s7.square_law, 0.4, color=C["grey"], label="square law")
    a.set_xticks(x, s7.ratio, rotation=90, fontsize=6.3)
    a.set_ylabel("P(next death on larger side)")
    a.set_ylim(0, 0.62)
    a.legend(fontsize=6.3, loc="upper left", ncol=2)
    a.set_title("(a) next death || by local numbers, vs the square law")
    b = axs[1]
    t = fl[(fl.model == "Tullock fight") & (fl.param == "gamma_n / gamma_g")]
    for i, (_, r) in enumerate(t.iterrows()):
        b.errorbar(i, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=3, color=[C["orange"], C["blue"]][i])
    b.set_xticks(range(len(t)), ["tracked\n+V5", "tracked\nonly"], fontsize=6.5)
    b.set_xlim(-0.5, len(t) - 0.5)
    b.set_ylabel("$\\gamma_n/\\gamma_g$")
    b.set_title("(b) Tullock ratio || two count definitions")
    fi = fl.set_index(["model", "counts", "param"])
    tr = "tracked only"
    v5 = "tracked + V5 participants"
    for key, model, cnt, prm, nd in (("tulGn", "Tullock fight", tr, "gamma_n", 2), ("tulGg", "Tullock fight", tr, "gamma_g", 1),
                                     ("tulRatio", "Tullock fight", tr, "gamma_n / gamma_g", 3),
                                     ("tulGnVfive", "Tullock fight", v5, "gamma_n", 2), ("tulRatioVfive", "Tullock fight", v5, "gamma_n / gamma_g", 3),
                                     ("tulSide", "Tullock fight", tr, "gamma_side", 3),
                                     ("sSevenBn", "S7 next death", tr, "b_n (square law -1, linear 0)", 2),
                                     ("sSevenBnVfive", "S7 next death", v5, "b_n (square law -1, linear 0)", 2)):
        r = fi.loc[(model, cnt, prm)]
        N[key] = f(r.est, nd)
        N[key + "CI"] = ci(r.lo, r.hi, nd)
        N[key + "N"] = f"{int(r.n):,}"
    gn, gg = fi.loc[("Tullock fight", tr, "gamma_n"), "est"], fi.loc[("Tullock fight", tr, "gamma_g"), "est"]
    N["tulFiveFourGoldEq"] = f(np.expm1(gn * np.log(5 / 4) / gg), 1, pct=True)
    s = s7.set_index("ratio")
    N["sSevenFiveThree"] = f"{s.loc['5v3', 'p_victim_larger']:.3f}"
    N["sSevenFiveThreeSq"] = f"{s.loc['5v3', 'square_law']:.3f}"
    N["sSevenTwoOne"] = f"{s.loc['2v1', 'p_victim_larger']:.3f}"
    N["sSevenTwoOneSq"] = f"{s.loc['2v1', 'square_law']:.3f}"
    return save(fig, "fig12_fights")


if __name__ == "__main__":
    run_main(build)
