"""Fig. 5: commitment curves by gold state on the V layer (risk set after the spawn, pit radius 1500).
Rows: P(k = 0) (pit left empty) and mean k; columns: five objective groups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

GROUPS = ["DRAGON_2+", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR", "BARON_2+"]


def build(N):
    cv = csv("results/v_layer/curves_by_state.csv")
    cv = cv[cv.R == 1500]
    fig, axs = plt.subplots(2, len(GROUPS), figsize=(TEXT_W, 3.0), sharex=True, gridspec_kw=dict(hspace=0.3, wspace=0.42))
    for j, gname in enumerate(GROUPS):
        for i, stat in enumerate(("P(k=0)", "mean k")):
            a = axs[i, j]
            for st in ("trailing", "even", "leading"):
                d = cv[(cv.group == gname) & (cv.stat == stat) & (cv.state == st) & (cv.n >= 30)].sort_values("offset_s")
                a.plot(d.offset_s, d.est, color=STATE_C[st], label=st, lw=1.0)
                a.fill_between(d.offset_s, d.lo, d.hi, color=STATE_C[st], alpha=0.15, lw=0)
            a.axvline(0, color="k", lw=0.5, ls=":")
            if i == 0:
                a.set_title(OBJ_LABEL[gname])
                a.set_ylim(0, 1.02)
            if j == 0:
                a.set_ylabel("P(pit empty)" if i == 0 else "mean $k$ at pit")
            if i == 1:
                a.set_xlabel("s from spawn")
            a.set_xticks([-120, 0, 120])
            if j > 0:
                a.set_ylabel("")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), fontsize=6.8)
    return save(fig, "fig05_curves")


if __name__ == "__main__":
    run_main(build)
