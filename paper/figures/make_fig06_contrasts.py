"""Fig. 6: trailing-minus-leading contrasts of P(k = 0) on the V layer across objective groups and time
(R = 1500). Colour = contrast; dots = headline cells (sign agrees at R = 1000/1500/2000 and BH q < 0.05)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

GROUPS = ["DRAGON", "DRAGON_2+", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR", "BARON_2+", "ELDER"]


def build(N):
    con = csv("results/v_layer/contrasts_trailing_minus_leading.csv")
    clean = csv("results/v_layer/contrasts_trailing_minus_leading_clean.csv")
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 2.5), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1]))
    for a, stat, lab, vmax in ((axs[0], "P(k=0)", "empty pit || trailing minus leading", 0.4),
                               (axs[1], "P(self >= 3)", "three or more at the pit || trailing minus leading", 0.15)):
        d = con[(con.R == 1500) & (con.stat == stat)]
        offs = sorted(d.offset_s.unique())
        M = np.full((len(GROUPS), len(offs)), np.nan)
        H = np.zeros_like(M, bool)
        for i, g in enumerate(GROUPS):
            for j, o in enumerate(offs):
                r = d[(d.group == g) & (d.offset_s == o)]
                if len(r):
                    M[i, j] = r.trailing_minus_leading.iloc[0]
                    H[i, j] = bool(r.headline.iloc[0])
        im = a.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        yy, xx = np.where(H)
        a.scatter(xx, yy, s=4, c="k")
        a.set_yticks(range(len(GROUPS)), [OBJ_LABEL[g] for g in GROUPS])
        step = 4
        a.set_xticks(range(0, len(offs), step), [str(o) for o in offs[::step]])
        a.set_xlabel("s from spawn")
        a.axvline(offs.index(0), color="k", lw=0.5, ls=":")
        cb = fig.colorbar(im, ax=a, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=6.3)
        a.set_title(lab)
    heads = con[con.headline & (con.R == 1500)]
    less = ((heads.stat == "P(k=0)") & (heads.trailing_minus_leading > 0)) | ((heads.stat != "P(k=0)") & (heads.trailing_minus_leading < 0))
    N["vHeadlineTrailLess"] = f"{int(less.sum())}"
    N["vHeadlineCells"] = f"{len(heads)}"
    N["vContrastCells"] = f"{len(con[con.R == 1500].dropna(subset=['trailing_minus_leading']))}"
    both = con.merge(clean, on=["group", "offset_s", "R", "stat"], suffixes=("", "_c"))
    hb = both[both.headline & (both.R == 1500)]
    N["vHeadlineCleanKept"] = f"{int(hb.headline_c.sum())}"
    N["vHeadlineCleanSign"] = f"{int((np.sign(hb.trailing_minus_leading) == np.sign(hb.trailing_minus_leading_c)).sum())}"

    def cell(g, o, s):
        r = con[(con.group == g) & (con.offset_s == o) & (con.R == 1500) & (con.stat == s)].iloc[0]
        return r
    for key, g, o, s in (("AtakhanFortyfive", "ATAKHAN", 45, "P(k=0)"), ("BaronFortyfive", "BARON_NASHOR", 45, "P(k=0)"),
                         ("BaronSixty", "BARON_NASHOR", 60, "P(k=0)"), ("DragonTwoFifteen", "DRAGON_2+", 15, "P(k=0)"),
                         ("DragonTwoFifteenSelf", "DRAGON_2+", 15, "P(self >= 3)"), ("HeraldMinusSeventyfive", "RIFTHERALD", -75, "P(k=0)"),
                         ("AtakhanFortyfiveMeanK", "ATAKHAN", 45, "mean k")):
        r = cell(g, o, s)
        nd = 2 if s == "mean k" else 1
        pct = s != "mean k"
        N[f"vC{key}"] = f(r.trailing_minus_leading, nd, pct, sign=True)
        N[f"vC{key}CI"] = ci(r.lo, r.hi, nd, pct)
    return save(fig, "fig06_contrasts")


if __name__ == "__main__":
    run_main(build)
