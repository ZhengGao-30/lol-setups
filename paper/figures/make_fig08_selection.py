"""Fig. 8: HASSE model selection. (a) held-out log-likelihood per setup vs K (V layer, 5 series-grouped folds),
(b) BIC and ICL vs K, (c) the same criteria on the robustness layers (7-snapshot grid, formation radius
1500 / 2500) and the R layer (relative to their own K = 3 value)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    v = csv("results/hasse_v/selection.csv").sort_values("K")
    r = csv("results/hasse/selection.csv").sort_values("K")
    rob = {k: csv(f"results/{k}/selection.csv").sort_values("K") for k in ("hasse_v7", "hasse_vr1500", "hasse_vr2500")}
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.3), gridspec_kw=dict(wspace=0.6))
    a = axs[0]
    a.plot(v.K, v.test_loglik_per_setup, "o-", ms=3, color=C["blue"])
    a.set_xlabel("$K$ (patterns)")
    a.set_ylabel("held-out log-lik / setup")
    a.axvline(6, color=C["grey"], lw=0.6, ls=":")
    a.set_title("(a) held-out fit || V layer, five folds")
    b = axs[1]
    b.plot(v.K, v.bic / 1e3, "o-", ms=3, color=C["blue"], label="BIC")
    b.plot(v.K, v.icl / 1e3, "s--", ms=3, color=C["orange"], label="ICL")
    b.set_xlabel("$K$")
    b.set_ylabel("criterion ($\\times 10^3$)")
    b.legend(fontsize=6.3)
    b.set_title("(b) BIC and ICL || V layer, full data")
    c = axs[2]
    lab = {"hasse_v7": "7 snapshots", "hasse_vr1500": "radius 1500", "hasse_vr2500": "radius 2500"}
    for (k, d), col in zip(rob.items(), (C["green"], C["purple"], C["sky"])):
        base = d.set_index("K").test_loglik_per_setup
        c.plot(base.index, base - base.loc[3], "o-", ms=2.5, color=col, label=lab[k])
    rb = r.set_index("K").test_loglik_per_setup
    c.plot(rb.index, rb - rb.loc[3], "^-", ms=2.5, color="k", lw=0.8, label="R layer")
    c.set_xlabel("$K$")
    c.set_ylabel("$\\Delta$ held-out vs $K{=}3$")
    c.legend(fontsize=6.0, loc="lower right")
    c.set_title("(c) robustness || relative to K = 3")
    vi = v.set_index("K")
    for K in range(1, 9):
        N[f"hvTestK{K}"] = f(vi.loc[K, "test_loglik_per_setup"], 2)
        N[f"hvBicK{K}"] = f"{vi.loc[K, 'bic']:,.0f}"
    N["hvStepFiveSix"] = f"{vi.loc[6, 'test_loglik_per_setup'] - vi.loc[5, 'test_loglik_per_setup']:.2f}"
    N["hvStepSixSeven"] = f"{vi.loc[7, 'test_loglik_per_setup'] - vi.loc[6, 'test_loglik_per_setup']:.2f}"
    N["hvStepSevenEight"] = f"{vi.loc[8, 'test_loglik_per_setup'] - vi.loc[7, 'test_loglik_per_setup']:.2f}"
    ri = r.set_index("K")
    N["hrBestBic"] = f"{int(ri.bic.idxmin())}"
    N["hrBestIcl"] = f"{int(ri.icl.idxmin())}"
    N["hrBestTest"] = f"{int(ri.test_loglik_per_setup.idxmax())}"
    for k, d in rob.items():
        di = d.set_index("K")
        N[f"best{k.replace('hasse_', '').replace('v7', 'Vseven').replace('vr1500', 'Rfifteen').replace('vr2500', 'Rtwentyfive')}"] = \
            f"{int(di.test_loglik_per_setup.idxmax())}"
    return save(fig, "fig08_selection")


if __name__ == "__main__":
    run_main(build)
