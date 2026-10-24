"""Fig. 13: LB-QRE. (a) the commitment gap P(k>=3 | trailing) - P(k>=3 | leading): observed, full model and all
channels off (300 series-block bootstrap replicates); (b) Shapley decomposition of the model gap into four
channels (bootstrap distributions); (c) the H_winmax test: log-likelihood loss from fixing nu = 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    reps = csv("results/robustness/qre_bootstrap_reps.csv")
    cib = csv("results/robustness/qre_bootstrap_ci.csv").set_index("quantity")
    pc = csv("results/robustness/qre_shapley_paired_contrasts.csv").set_index("contrast")
    var = csv("results/robustness/qre_variants.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.4), gridspec_kw=dict(wspace=0.62, width_ratios=[0.9, 1.3, 1]))
    a = axs[0]
    for i, (q, lab, col) in enumerate((("observed_gap", "observed", "k"), ("gap_full", "full model", C["blue"]),
                                        ("gap_none", "channels off", C["grey"]))):
        r = cib.loc[q]
        a.errorbar(i, 100 * r.point, yerr=[[100 * (r.point - r.lo)], [100 * (r.hi - r.point)]], fmt="o", ms=3.5, color=col)
    a.set_xticks(range(3), ["observed", "model", "off"], fontsize=6.5)
    a.axhline(0, color="k", lw=0.5)
    a.set_xlim(-0.5, 2.5)
    a.set_ylabel("commitment gap (pp)")
    a.set_title("(a) commitment gap || trailing minus leading")
    b = axs[1]
    chans = [("shap_L", "Lanchester"), ("shap_C", "curvature"), ("shap_F", "second front"), ("shap_B", "behavioural")]
    parts = b.violinplot([100 * reps[c] for c, _ in chans], vert=False, showmedians=True, widths=0.8)
    for pc_, col in zip(parts["bodies"], [C["orange"], C["blue"], C["green"], C["red"]]):
        pc_.set_facecolor(col)
        pc_.set_alpha(0.55)
    for i, (c, _) in enumerate(chans):
        b.scatter(100 * var[c], np.full(len(var), i + 1) + 0.3, s=5, marker="|", color="k", lw=0.6)
    b.set_yticks(range(1, 5), [l for _, l in chans], fontsize=6.5)
    b.axvline(0, color="k", lw=0.5)
    b.set_xlabel("Shapley value (pp of the gap)")
    b.set_title("(b) four channels || Shapley, ticks = variants")
    c = axs[2]
    c.hist(reps.dll_winmax.dropna(), bins=25, color=C["orange"])
    c.set_xlabel("log-lik loss of $\\nu=1$")
    c.set_ylabel("bootstrap replicates")
    c.set_title("(c) win-maximisation || rejected in every replicate")
    for q, key, nd, pct in (("shap_L", "shL", 2, True), ("shap_C", "shC", 2, True), ("shap_F", "shF", 2, True),
                            ("shap_B", "shB", 2, True), ("gap_full", "gapModel", 1, True), ("gap_none", "gapNone", 1, True),
                            ("observed_gap", "gapObs", 1, True), ("lam", "qLam", 2, False), ("nu", "qNu", 3, False),
                            ("kappa", "qKappa", 2, False), ("psi", "qPsi", 2, False), ("omega_k", "qOmegaK", 3, False),
                            ("omega_m", "qOmegaM", 2, False), ("dll_winmax", "qWinmax", 1, False)):
        r = cib.loc[q]
        pt = r.point if not np.isnan(r.point) else r.boot_mean
        N[key] = f(pt, nd, pct, sign=key.startswith("sh") or key.startswith("gap"))
        N[key + "CI"] = ci(r.lo, r.hi, nd, pct)
    N["qLamObj"] = f"{cib.loc['lam', 'point'] * cib.loc['nu', 'point']:.3f}"
    N["qLamOnePoint"] = f(np.expm1(cib.loc["lam", "point"] * 0.01), 1, pct=True)
    N["qReps"] = f"{int(cib.n_reps.iloc[0])}"
    N["shCminusL"] = f(pc.loc["C-F", "mean"] if False else -pc.loc["L-C", "mean"], 2, True, sign=True)
    N["shCminusLCI"] = ci(-pc.loc["L-C", "hi"], -pc.loc["L-C", "lo"], 2, True)
    N["gapShare"] = f"{cib.loc['gap_full', 'point'] / cib.loc['observed_gap', 'point']:.2f}"
    m = var[var.variant == "(main)"].iloc[0]
    N["qContests"] = f"{int(m.n_contests):,}"
    N["qLL"] = f(m.ll_qre, 0)
    N["qLLsat"] = f(m.ll_saturated, 0)
    N["varShapMax"] = f(var[["shap_L", "shap_C", "shap_F", "shap_B"]].abs().max().max(), 1, pct=True)
    sub = var[var.variant.isin(["_r1000", "_r2000"])].set_index("variant")
    N["varObsGapRone"] = f(sub.loc["_r1000", "observed_gap"], 2)
    N["varObsGapRtwo"] = f(sub.loc["_r2000", "observed_gap"], 2)
    nc = var[var.variant == "_nocost"].iloc[0]
    N["nocostLL"] = f(nc.ll_qre, 0)
    N["holdContests"] = f"{int(var[var.variant == '_hold'].n_contests.iloc[0])}"
    wm = var[var.variant == "_winmax"].iloc[0]
    N["qWinmaxPoint"] = f(m.ll_qre - wm.ll_qre, 1)
    N["qWinmaxLR"] = f(2 * (m.ll_qre - wm.ll_qre), 0)
    N["qWinmaxMin"] = f(reps.dll_winmax.min(), 1)
    cv = reps.converged.astype(bool)
    N["qConvN"] = f"{int(cv.sum())}"
    N["qConvShare"] = f(cv.mean(), 0, pct=True)
    N["qNuConv"] = f(reps.loc[cv, "nu"].mean(), 3)
    N["qNuNonconv"] = f(reps.loc[~cv, "nu"].mean(), 3)
    sh = ["shap_L", "shap_C", "shap_F", "shap_B"]
    N["qConvShapDiff"] = f((reps.loc[cv, sh].mean() - reps.loc[~cv, sh].mean()).abs().max(), 1, pct=True)
    N["gapChannels"] = f(cib.loc["gap_full", "point"] - cib.loc["gap_none", "point"], 1, pct=True, sign=True)
    N["gapChannelsAbs"] = f(abs(cib.loc["gap_full", "point"] - cib.loc["gap_none", "point"]), 1, pct=True)
    return save(fig, "fig13_qre")


if __name__ == "__main__":
    run_main(build)
