"""Fig. 14: team fingerprints with intervals. (a) pattern usage (K = 6) per team with series-bootstrap
intervals, T1 highlighted; (b) split-half reliability of usage per pattern; (c) team decision precision lambda with
95% intervals (log scale)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    u = csv("results/fingerprints_K6/usage_ci.csv")
    sh = csv("results/fingerprints_K6/split_half.csv")
    tp = csv("results/robustness/qre_team_parameter_ci.csv")
    nb = csv("results/fingerprints_K6/neighbours_T1.csv").rename(columns={"Unnamed: 0": "team"})
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.7), gridspec_kw=dict(wspace=0.62, width_ratios=[1.5, 0.8, 1.1]))
    a = axs[0]
    teams = u.drop_duplicates("team").sort_values("n_setups", ascending=False).team.tolist()[:10]
    for j, t in enumerate(teams):
        g = u[u.team == t].sort_values("pattern")
        col = C["red"] if t == "T1" else C["grey"]
        x = g.pattern + (j - len(teams) / 2) * 0.07
        a.errorbar(x, g.usage, yerr=[g.usage - g.lo, g.hi - g.usage], fmt="o", ms=2 if t != "T1" else 3,
                   color=col, lw=0.5 if t != "T1" else 1.0, alpha=1 if t == "T1" else 0.6,
                   label="T1" if t == "T1" else ("other teams" if j == 1 else None))
    a.set_xticks(range(6), [f"P{i}" for i in range(6)])
    a.set_ylabel("pattern usage")
    a.legend(fontsize=6.3)
    a.set_title("(a) usage || ten largest teams")
    b = axs[1]
    b.errorbar(sh.corr_mean, sh.pattern, xerr=[sh.corr_mean - sh.corr_lo, sh.corr_hi - sh.corr_mean], fmt="o", ms=3, color=C["blue"])
    b.axvline(0, color="k", lw=0.5)
    b.set_yticks(range(6), [f"P{i}" for i in range(6)])
    b.invert_yaxis()
    b.set_xlabel("split-half corr.")
    b.set_title("(b) reliability || split-half")
    c = axs[2]
    lam = tp[(tp.param == "lam") & (tp.team != "other")].sort_values("point")
    y = np.arange(len(lam))
    cols = [C["red"] if t == "T1" else C["blue"] for t in lam.team]
    c.errorbar(lam.point, y, xerr=[lam.point - lam.lo, lam.hi - lam.point], fmt="none", ecolor=cols, lw=0.7)
    c.scatter(lam.point, y, s=6, c=cols, zorder=3)
    c.set_xscale("log")
    c.set_yticks(y, lam.team, fontsize=5.8)
    c.set_xlabel("decision precision $\\lambda_i$")
    c.set_title("(c) precision || LB-QRE, 95% intervals")
    N["shMin"] = f(sh.corr_mean.min(), 2)
    N["shMax"] = f"{sh.corr_mean.max():.2f}"
    N["shZeroIn"] = f"{int(((sh.corr_lo < 0) & (sh.corr_hi > 0)).sum())}"
    nn = u.drop_duplicates("team").n_setups
    N["setupsTeamMin"], N["setupsTeamMax"] = f"{int(nn.min())}", f"{int(nn.max())}"
    t1 = u[u.team == "T1"].sort_values("pattern")
    N["tOneSetups"] = f"{int(t1.n_setups.iloc[0])}"
    N["tOneUsage"] = "/".join(f"{x:.2f}" for x in t1.usage)
    nb = nb.sort_values("tv_to_T1")
    N["tOneNearest"] = nb.team.iloc[0]
    N["tOneNearestTV"] = f"{nb.tv_to_T1.iloc[0]:.3f}"
    N["tOneNearestP"] = f(nb.p_nearest_to_T1.iloc[0], 0, pct=True)
    N["tOneSecond"], N["tOneThird"] = nb.team.iloc[1], nb.team.iloc[2]
    tl = tp.set_index(["team", "param"])
    for t, k in (("T1", "TOne"), ("Gen.G", "GenG")):
        N[f"lam{k}"] = f"{tl.loc[(t, 'lam'), 'point']:.1f}"
        N[f"lam{k}CI"] = ci(tl.loc[(t, "lam"), "lo"], tl.loc[(t, "lam"), "hi"], 1)
    lo_obj = tl.xs("lam_obj", level="param") if "lam_obj" in tp.param.unique() else None
    if lo_obj is not None:
        N["lamObjTOne"] = f"{lo_obj.loc['T1', 'point']:.3f}"
        N["lamObjTOneCI"] = ci(lo_obj.loc["T1", "lo"], lo_obj.loc["T1", "hi"], 3)
    return save(fig, "fig14_fingerprints")


if __name__ == "__main__":
    run_main(build)
