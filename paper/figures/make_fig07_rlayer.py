"""Fig. 7: the R layer (exact Riot frames). (a) first-baron P(k = 0) by gold state at s and s + 60 for the
three pit radii; (b) the estimator audit: the same trailing-minus-leading quantity under three estimators;
(c) the pilot dragon gradient (kill-anchored, outcome-selected) vs the spawn-anchored risk-set contrast."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def parse(s):
    m = re.match(r"\s*(-?[\d.]+)%\s*\[(-?[\d.]+)%,\s*(-?[\d.]+)%\](?:\s*\(p=([\d.]+)\))?", s)
    return tuple(float(x) if x else np.nan for x in m.groups())


def build(N):
    al = csv("results/r_layer/allocation_by_state.csv")
    con = csv("results/r_layer/contrasts_trailing_minus_leading.csv")
    aud = csv("results/r_layer/estimator_audit.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.5), gridspec_kw=dict(wspace=0.7, width_ratios=[1.25, 1, 1.05]))
    a = axs[0]
    d = al[(al.objective == "BARON_NASHOR") & (al.stat == "P(k=0)")]
    xs, labels = [], []
    for i, (off, R) in enumerate([(o, r) for o in (0, 60) for r in (1000, 1500, 2000)]):
        for j, st in enumerate(("trailing", "even", "leading")):
            r = d[(d.offset_s == off) & (d.R == R) & (d.state == st)]
            if len(r):
                r = r.iloc[0]
                x = i + (j - 1) * 0.22
                a.errorbar(x, r.est, yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=2.5, color=STATE_C[st],
                           lw=0.8, label=st if i == 0 else None)
        xs.append(i)
        labels.append(f"{'s' if off == 0 else 's+60'}\n{R}")
    a.set_xticks(xs, labels, fontsize=6.3)
    a.set_ylabel("P(baron pit empty)")
    a.legend(fontsize=6.3, loc="lower left")
    a.set_title("(a) first baron || exact Riot frames, by gold state")

    b = axs[1]
    rows = aud[aud.quantity == "trailing - leading"]
    names = ["TAGNet-style\nteam mean", "pooled", "within-team"]
    for i, (_, r) in enumerate(rows.iterrows()):
        e, lo, hi, p = parse(r.value)
        b.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", ms=3, color=[C["red"], C["blue"], C["grey"]][i])
        b.text(hi + 0.4, i, f"p={p:.3f}", va="center", fontsize=6.3)
        key = ["Tagnet", "Pooled", "Within"][i]
        N[f"audit{key}"] = f(e, 1, sign=True)
        N[f"audit{key}CI"] = ci(lo, hi, 1)
        N[f"audit{key}P"] = f"{p:.3f}"
    b.axvline(0, color="k", lw=0.5)
    b.set_yticks(range(3), names, fontsize=6.5)
    b.invert_yaxis()
    b.set_xlabel("$\\Delta$ P($\\geq$3 at pit), pp")
    b.set_xlim(-13, 6)
    b.set_title("(b) estimator audit || one pooled contrast")

    c = axs[2]
    pil = pd.read_csv(src("results/pilot_objective_setups/non_taker_contest_by_gold_state.csv"), header=[0, 1], index_col=0)
    pdrag = pil.loc["DRAGON"]
    gap_pilot = pdrag[("mean", "trailing")] - pdrag[("mean", "leading")]
    rc = con[(con.objective == "DRAGON") & (con.offset_s == 60) & (con.R == 1500)]
    vals = [("pilot\n(kill-anchored)", gap_pilot, np.nan, np.nan, C["grey"])]
    for stat, lab in (("P(self >= 3)", "risk set\n$\\geq$3 at pit"), ("P(k=0)", "risk set\nempty pit")):
        r = rc[rc.stat == stat]
        if len(r):
            r = r.iloc[0]
            vals.append((lab, r.trailing_minus_leading, r.lo, r.hi, C["blue"]))
    for i, (lab, e, lo, hi, col) in enumerate(vals):
        c.bar(i, 100 * e, color=col, width=0.6)
        if not np.isnan(lo):
            c.errorbar(i, 100 * e, yerr=[[100 * (e - lo)], [100 * (hi - e)]], color="k", lw=0.8, capsize=2)
    c.axhline(0, color="k", lw=0.5)
    c.set_xticks(range(len(vals)), [v[0] for v in vals], fontsize=6.5)
    c.set_ylabel("trailing $-$ leading (pp)")
    c.set_title("(c) first dragon || pilot vs risk set")
    d0 = al[(al.objective == "DRAGON") & (al.offset_s == 0) & (al.R == 1500) & (al.stat == "P(k=0)")]
    N["rDragonEmptyAtSpawnMin"] = f(d0.est.min(), 0, pct=True)
    N["pilotDragonTrail"] = f(pdrag[("mean", "trailing")], 0, pct=True)
    N["pilotDragonLead"] = f(pdrag[("mean", "leading")], 0, pct=True)
    N["pilotDragonLeadN"] = f"{int(pdrag[('count', 'leading')])}"
    r = rc[rc.stat == "P(k=0)"].iloc[0]
    N["rDragonSixtyEmpty"] = f(r.trailing_minus_leading, 1, pct=True, sign=True)
    N["rDragonSixtyEmptyCI"] = ci(r.lo, r.hi, 1, pct=True)
    b60 = con[(con.objective == "BARON_NASHOR") & (con.offset_s == 60) & (con.R == 1500)].set_index("stat")
    N["rBaronEmpty"] = f(b60.loc["P(k=0)", "trailing_minus_leading"], 1, pct=True, sign=True)
    N["rBaronEmptyCI"] = ci(b60.loc["P(k=0)", "lo"], b60.loc["P(k=0)", "hi"], 1, pct=True)
    N["rBaronMeanK"] = f(b60.loc["mean k", "trailing_minus_leading"], 2, sign=True)
    N["rBaronMeanKCI"] = ci(b60.loc["mean k", "lo"], b60.loc["mean k", "hi"], 2)
    N["rHeadlines"] = f"{int(con[con.headline].groupby(['objective', 'offset_s', 'stat']).ngroups)}"
    at = con[(con.objective == "ATAKHAN") & (con.offset_s == 60) & (con.stat == "P(k=0)")]
    N["rAtakhanEmptyLo"] = f(at.trailing_minus_leading.min(), 0, pct=True)
    N["rAtakhanEmptyHi"] = f(at.trailing_minus_leading.max(), 0, pct=True)
    N["rAtakhanQ"] = f"{at.q_bh.min():.2f}"
    return save(fig, "fig07_rlayer")


if __name__ == "__main__":
    run_main(build)
