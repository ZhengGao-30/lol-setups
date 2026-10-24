"""Appendix figures on the models: W regularisation path, TAGNet K selection, context AMEs at K = 5 and 8,
QRE specification variants, and all team cost parameters with intervals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403
import importlib.util  # noqa: E402


def build(N):
    out = []
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W * 0.75, 2.0), gridspec_kw=dict(wspace=0.6))
    for k, col in (("pooled", C["blue"]), ("r2509", C["orange"]), ("all", C["green"])):
        cv = csv(f"results/winsurface/{k}/cv_penalty.csv")
        axs[0].plot(cv.C, cv.logloss, "o-", ms=2, color=col, label=k)
        axs[1].plot(cv.C, cv.brier, "o-", ms=2, color=col)
    for a, y in zip(axs, ("CV log-loss", "CV Brier")):
        a.set_xscale("log")
        a.set_xlabel("inverse penalty $C$")
        a.set_ylabel(y)
    axs[0].legend(fontsize=6.3)
    out.append(save(fig, "figA_w_cv"))

    tk = csv("results/baseline/tagnet_k_selection.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.0), gridspec_kw=dict(wspace=0.55))
    for a, m, lab in zip(axs, ("silhouette", "dbi", "ch"), ("silhouette", "Davies–Bouldin", "Calinski–Harabasz")):
        for v, col in (("noctx", C["grey"]), ("ctx", C["red"])):
            d = tk[tk.variant == v]
            a.plot(d.K, d[m], "o-", ms=2, color=col, label="TAGNet" + (" + ctx" if v == "ctx" else ""))
        a.set_xlabel("$K$")
        a.set_title(lab)
    axs[0].legend(fontsize=6.3)
    out.append(save(fig, "figA_tagnet_k"))
    N["tgSilSix"] = f"{tk[(tk.variant == 'noctx') & (tk.K == 6)].silhouette.iloc[0]:.3f}"

    spec = importlib.util.spec_from_file_location("ctx", Path(__file__).with_name("make_fig11_context.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for K in (5, 8):
        try:
            out.append(mod.build(N, K=K, name=f"figA_context_K{K}"))
        except Missing:
            pass

    var = csv("results/robustness/qre_variants.csv")
    lab = {"(main)": "main", "_hold": "drop held", "_law_v5": "V5 fight law", "_nocost": "no costs (orig.)",
           "_nocost_law_v5": "no costs + V5", "_r1000": "pit R=1000", "_r2000": "pit R=2000", "_result": "'result' values",
           "_winmax": "$\\nu=1$"}
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 2.6), gridspec_kw=dict(wspace=0.75))
    y = np.arange(len(var))
    axs[0].scatter(100 * var.observed_gap, y, color="k", s=8, label="observed")
    axs[0].scatter(100 * var.model_gap, y, color=C["blue"], s=8, label="model")
    axs[0].set_yticks(y, [lab.get(v, v) for v in var.variant], fontsize=6.5)
    axs[0].invert_yaxis()
    axs[0].axvline(0, color="k", lw=0.5)
    axs[0].set_xlabel("commitment gap (pp)")
    axs[0].legend(fontsize=6.3)
    for i, (c, col) in enumerate((("shap_L", C["orange"]), ("shap_C", C["blue"]), ("shap_F", C["sky"]), ("shap_B", C["red"]))):
        axs[1].barh(y + (i - 1.5) * 0.2, 100 * var[c], 0.2, color=col, label=c[-1])
    axs[1].set_yticks(y, [lab.get(v, v) for v in var.variant], fontsize=6.5)
    axs[1].invert_yaxis()
    axs[1].axvline(0, color="k", lw=0.5)
    axs[1].set_xlabel("Shapley value (pp)")
    axs[1].legend(fontsize=6.3, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    out.append(save(fig, "figA_qre_variants"))

    tp = csv("results/robustness/qre_team_parameter_ci.csv")
    tp = tp[tp.team != "other"]
    params = [p for p in ("nu", "kappa", "psi", "omega_k", "omega_m") if p in set(tp.param)]
    fig, axs = plt.subplots(1, len(params), figsize=(TEXT_W, 3.2), sharey=True, gridspec_kw=dict(wspace=0.12))
    order = tp[tp.param == "lam"].sort_values("point").team.tolist()
    for a, p in zip(axs, params):
        d = tp[tp.param == p].set_index("team").reindex(order)
        y = np.arange(len(d))
        cols = [C["red"] if t == "T1" else C["blue"] for t in d.index]
        a.errorbar(d.point, y, xerr=[d.point - d.lo, d.hi - d.point], fmt="none", ecolor=cols, lw=0.6)
        a.scatter(d.point, y, s=5, c=cols, zorder=3)
        a.set_title({"nu": "$\\nu$", "kappa": "$\\kappa$", "psi": "$\\psi$", "omega_k": "$\\omega_k$", "omega_m": "$\\omega_m$"}[p])
        a.tick_params(axis="x", labelsize=6.0)
    axs[0].set_yticks(np.arange(len(order)), order, fontsize=5.8)
    out.append(save(fig, "figA_team_params"))
    return out


if __name__ == "__main__":
    run_main(build)
