"""Main-text tables (all numbers read from result files)."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

OBJN = {"DRAGON": "Elemental dragon", "ELDER": "Elder dragon", "HORDE": "Void grubs", "RIFTHERALD": "Rift Herald",
        "ATAKHAN": "Atakhan", "BARON_NASHOR": "Baron Nashor"}
RULES = {"DRAGON": ("5:00", "kill + 5:00 until a soul"), "ELDER": ("soul + 6:00", "kill + 6:00"),
         "HORDE": ("8:00", "one camp of 3, no respawn"), "RIFTHERALD": ("15:00", "once"),
         "ATAKHAN": ("20:00", "once (side set by play)"), "BARON_NASHOR": ("25:00", "kill + 6:00")}


def p(s):
    m = re.match(r"\s*(-?[\d.]+)%\s*\[(-?[\d.]+)%,\s*(-?[\d.]+)%\](?:\s*\(p=([\d.]+)\))?", s)
    return [float(x) if x else np.nan for x in m.groups()]


def t_data(N):
    ev = csv("results/data/table1_events.csv")
    ob = csv("results/data/table1_objectives.csv").set_index("objective")
    ps = csv("results/data/position_sources_by_objective.csv").set_index("objective")
    rows = []
    for _, r in ev.iterrows():
        nm = str(r.event).replace("2025 Season World Championship/", "Worlds 2025 ").replace("2025 Mid-Season Invitational", "MSI 2025")
        rows.append([tex_escape(nm), f"{r.games}", f"{r.series}", f"{r.teams}", f"{r.spawns:,}"])
    write_table("tab_data", ["Event", "Games", "Series", "Teams", "Spawns"], rows, caption=
                "The core video set. Every game of MSI 2025 and Worlds 2025 under one set of objective rules.",
                label="tab:data", midrules=(len(rows) - 1,))
    tot = ev[ev.event == "total"].iloc[0]
    N["nGames"], N["nSeries"], N["nTeams"], N["nSpawns"] = f"{tot.games}", f"{tot.series}", f"{tot.teams}", f"{tot.spawns:,}"
    N["nSetups"] = f"{int(ob.team_setups.sum()):,}"
    rows = []
    for o in ["DRAGON", "ELDER", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR"]:
        r = ob.loc[o]
        s = ps.loc[o]
        rows.append([OBJN[o], RULES[o][0], RULES[o][1], f"{int(r.spawns):,}", f"{int(r.taken):,}", f"{r.median_delay_s:.0f}",
                     f"{int(r.team_setups):,}", f"{s.q_det:.2f}/{s.q_interp:.2f}/{s.q_pin:.2f}"])
        N[f"delay{o.title().replace('_', '')}"] = f"{r.median_delay_s:.0f}"
    write_table("tab_rules", ["Objective", "First spawn", "Respawn", "Spawns", "Taken", "Delay (s)", "Setups",
                              "det/int/pin"], rows, align="lllrrrrr", star=True, caption=
                "The spawn clock (patch 25.09--25.20 rules) and the resulting units. Delay is the median spawn-to-kill time; "
                "the last column gives the share of tracked positions that were detected, interpolated or pinned to a V5 minute.",
                label="tab:rules")


def t_tracker(N):
    pooled = csv("results/tracker/summary_pooled.csv").iloc[0]
    st = csv("results/tracker/summary_by_state.csv").set_index("state")
    cr = csv("results/tracker/summary_by_crowding.csv")
    rows = [["pooled", "100", f"{pooled.q50:,.0f}", f"{pooled.q90:,.0f}", f"{pooled.lt300:.2f}", f"{pooled.lt600:.2f}", f"{pooled.lt1000:.2f}"]]
    for s_, lab in (("det", "detected"), ("interp", "interpolated"), ("hold", "held")):
        r = st.loc[s_]
        rows.append([lab, f"{100 * r.share:.0f}", f"{r.q50:,.0f}", f"{r.q90:,.0f}", f"{r.lt300:.2f}", f"{r.lt600:.2f}", f"{r.lt1000:.2f}"])
    for _, r in cr.iterrows():
        rows.append([f"crowding {r.crowd_bin}", f"{100 * r.share:.0f}", f"{r.q50:,.0f}", f"{r.q90:,.0f}", f"{r.lt300:.2f}", f"{r.lt600:.2f}", f"{r.lt1000:.2f}"])
    write_table("tab_tracker", ["Subset", "\\%", "Median", "P90", "$<$300", "$<$600", "$<$1000"], rows, caption=
                f"Tracker error against Riot V5 positions at held-out odd minutes ({int(pooled.n):,} champion-minutes, 164 games; "
                "map units, map width 14,820). Crowding = other champions within 2,000 units.", label="tab:tracker",
                midrules=(1, 4))


def t_rlayer(N):
    con = csv("results/r_layer/contrasts_trailing_minus_leading.csv")
    aud = csv("results/r_layer/estimator_audit.csv")
    rows = []
    for o in ["DRAGON", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR"]:
        for stat in ("P(k=0)", "mean k"):
            r = con[(con.objective == o) & (con.offset_s == 60) & (con.R == 1500) & (con.stat == stat)]
            if not len(r) or np.isnan(r.trailing_minus_leading.iloc[0]):
                continue
            r = r.iloc[0]
            pct = stat != "mean k"
            rows.append([OBJN[o] if stat == "P(k=0)" else "", "pit empty" if pct else "mean $k$",
                         f(r.trailing_minus_leading, 1 if pct else 2, pct, sign=True), ci(r.lo, r.hi, 1 if pct else 2, pct),
                         f"{r.q_bh:.3f}", "\\checkmark" if r.headline else ""])
    n0 = len(rows)
    for _, r in aud[aud.quantity == "trailing - leading"].iterrows():
        e, lo, hi, pv = p(r.value)
        rows.append([tex_escape(r.estimator.replace("TAGNet-style unweighted team mean", "TAGNet-style team mean")), "$\\geq$3 at pit",
                     f(e, 1, sign=True), ci(lo, hi, 1), f"$p$={pv:.3f}", ""])
    write_table("tab_rlayer", ["Objective / estimator", "Quantity", "T$-$L (pp)", "95\\% CI", "$q_{BH}$", "Head."], rows,
                align="llrrrc", star=True, caption="R layer (exact Riot frames) at $s{+}60$, risk set, pit radius 1,500. Top: trailing minus "
                "leading by objective (headline = sign agrees at radii 1,000/1,500/2,000 and $q_{BH}<0.05$). Bottom: one pooled "
                "contrast under three estimators.", label="tab:rlayer", midrules=(n0,))


def t_vlayer(N):
    con = csv("results/v_layer/contrasts_trailing_minus_leading.csv")
    h = con[con.headline & (con.R == 1500)]
    rows = []
    for g in ["RIFTHERALD", "HORDE", "DRAGON", "DRAGON_2+", "ATAKHAN", "BARON_NASHOR", "BARON_2+"]:
        for stat in ("P(k=0)", "P(self >= 3)", "mean k"):
            d = h[(h.group == g) & (h.stat == stat)]
            if not len(d):
                continue
            i = d.trailing_minus_leading.abs().idxmax()
            r = d.loc[i]
            pct = stat != "mean k"
            rows.append([OBJ_LABEL[g], {"P(k=0)": "pit empty", "P(self >= 3)": "$\\geq$3 at pit", "mean k": "mean $k$"}[stat],
                         f"{len(d)}", f"$s{int(r.offset_s):+d}$", f(r.trailing_minus_leading, 1 if pct else 2, pct, sign=True),
                         ci(r.lo, r.hi, 1 if pct else 2, pct)])
    write_table("tab_vlayer", ["Objective", "Quantity", "Cells", "Peak at", "T$-$L", "95\\% CI"], rows, align="llrrrr",
                caption="V layer: headline trailing-minus-leading contrasts (sign agrees at three pit radii, $q_{BH}<0.05$). "
                "Cells = number of headline time points; the largest is shown. Units: pp for shares, players for mean $k$.",
                label="tab:vlayer")


def t_selection(N):
    v = csv("results/hasse_v/selection.csv").set_index("K").sort_index()
    r = csv("results/hasse/selection.csv").set_index("K").sort_index()
    rows = []
    for K in v.index:
        rr = r.loc[K] if K in r.index else None
        rows.append([f"{K}", f"{v.loc[K, 'n_params']:.0f}", f"{v.loc[K, 'test_loglik_per_setup']:.2f}", f"{v.loc[K, 'bic']:,.0f}",
                     f"{v.loc[K, 'icl']:,.0f}", f"{rr.test_loglik_per_setup:.2f}" if rr is not None else "--",
                     f"{rr.bic:,.0f}" if rr is not None else "--"])
    write_table("tab_selection", ["$K$", "Params", "Held-out/setup", "BIC", "ICL", "R: held-out", "R: BIC"], rows,
                caption="HASSE model selection. V layer (13 snapshots, 2,994 setups) and R layer (3 exact frames); held-out "
                "log-likelihood per setup over five series-grouped folds.", label="tab:selection")


def t_patterns(N):
    j = js("results/hasse_v/patterns_K6.json")
    rows = []
    for p_ in j["patterns"]:
        cp = [c.replace(" ", "") for c in p_["centre_path"]]
        w = p_["weight_by_class"]
        rows.append([f"P{p_['pattern']}", f"{p_['theta']:.2f}", f"{p_['iota']:.2f}", f"{w[0]:.2f}", f"{w[1]:.2f}", f"{w[2]:.2f}",
                     f"\\texttt{{{cp[2]}}}", f"\\texttt{{{cp[6]}}}", f"\\texttt{{{cp[8]}}}", f"\\texttt{{{cp[12]}}}"])
    write_table("tab_patterns", ["", "$\\theta$", "$\\iota$", "$w_{river}$", "$w_{Atak.}$", "$w_{late}$", "$s{-}90$", "$s{-}30$",
                                 "$s$", "$s{+}60$"], rows, align="lrrrrrllll", star=True,
                caption=f"The pattern dictionary ($K=6$, V layer). Centre path $\\mu_{{c,t}}$ at four of the 13 snapshots "
                "(T top, J jungle, M mid, B bot, S support; bars separate blocks), rigidity $\\theta$ and inertia $\\iota$ "
                f"(per Hasse step) and the mixture weight in each objective class. Switching probability $\\xi={j['xi']:.3f}$.",
                label="tab:patterns")


def t_hasse_tagnet(N):
    s5, s6 = js("results/hasse_v/stability_K5.json"), js("results/hasse_v/stability_K6.json")
    tg = js("results/baseline/tagnet_summary.json")
    lk = csv("results/context/leakage_control_K5.csv").set_index("model")
    mono = js("results/hasse_vmono/mono_compare_K5.json")
    rows = [
        ["Setups used", "2,994 (all)", f"{tg['n_setups']:,} ($\\geq$6 of 7 pre-spawn)"],
        ["Choice of $K$", "held-out lik., BIC, ICL", "fixed $K=6$ (K-Means)"],
        ["Repeat-fit ARI", f"{s6['cross_restart_ari_mean']:.3f} (restarts)", f"{tg['noctx_pairwise_ari']:.3f} / {tg['ctx_pairwise_ari']:.3f} (seeds)"],
        ["Series-bootstrap ARI", f"{s6['bootstrap_ari_mean']:.3f} (q05 {s6['bootstrap_ari_q05']:.3f}; whole model)",
         f"{tg['noctx_bootstrap_ari_mean']:.3f} (q05 {tg['noctx_bootstrap_ari_q05']:.3f}; K-Means only)"],
        ["Agreement with counting baseline", "--", f"ARI {tg['count_baseline_vs_tagnet_ari']:.3f}"],
        ["Cramér's V with gold (clean)", f"{lk.cramers_V_gold.iloc[0]:.3f}", f"{lk.cramers_V_gold.iloc[2]:.3f}"],
        ["Cramér's V with gold (context fed in)", f"{lk.cramers_V_gold.iloc[1]:.3f}", f"{lk.cramers_V_gold.iloc[3]:.3f} ($\\times${tg['leakage_inflation_V_gold']:.2f})"],
        ["Parameters with units", "$\\theta,\\iota$ (Hasse steps), $\\xi$, zones", "64-d embedding"],
        ["`only gather' test $H_{mono}$", f"$\\Delta$ held-out {mono['heldout_diff_per_setup_mean']:.3f}/setup ({mono['folds_favouring_unrestricted']}/{mono['n_folds']} folds)", "--"],
    ]
    write_table("tab_hasse_tagnet", ["", "HASSE ($K=6$)", "TAGNet port"], rows, align="lll", star=True,
                caption="HASSE against TAGNet ported to the same V-layer tracks (TAGNet uses the 2,510 setups with at least 6 of 7 pre-spawn snapshots). HASSE rows are five independent fits, each the best of ten restarts; TAGNet rows are ten single runs per variant, with and without its context vector.",
                label="tab:hassetagnet")
    N["monoDiff"] = f"{mono['heldout_diff_per_setup_mean']:.3f}"
    N["monoSD"] = f"{mono['heldout_diff_per_setup_sd_over_folds']:.3f}"
    N["monoFolds"] = f"{mono['folds_favouring_unrestricted']}"
    N["monoNats"] = f"{mono['loglik_unrestricted'] - mono['loglik_mono']:.0f}"


def t_fights(N):
    fl = csv("results/fights/fight_laws.csv")
    rows = []
    lab = {"c": "intercept", "b_n (square law -1, linear 0)": "$b_n$ (square $-1$, linear 0)", "b_g": "$b_g$ (gold)",
           "gamma_n": "$\\gamma_n$ (numbers)", "gamma_g": "$\\gamma_g$ (gold)", "gamma_side": "$\\gamma_{side}$",
           "gamma_n / gamma_g": "$\\gamma_n/\\gamma_g$"}
    for _, r in fl.iterrows():
        rows.append([r.model.replace("S7 next death", "Next death"), "tracked" if r.counts == "tracked only" else "tracked+V5", lab.get(r.param, r.param),
                     f(r.est, 3), ci(r.lo, r.hi, 3), f"{int(r.n):,}"])
    write_table("tab_fights", ["Model", "Counts", "Parameter", "Est.", "95\\% CI", "$n$"], rows, align="lllrrr", star=True,
                caption="Fight laws. Next death: logit of the next death falling on the larger side on $\\ln(n_{large}/n_{small})$ and "
                "gold; Tullock: fight win on $\\ln(k_i/k_j)$ and $\\ln(\\bar g_i/\\bar g_j)$. Counts within 2,000 units 1 s before "
                "a kill; `tracked+V5' adds V5 killers and assisters (endogenous; sensitivity).", label="tab:fights",
                midrules=(7,))


def t_qre(N):
    cib = csv("results/robustness/qre_bootstrap_ci.csv").set_index("quantity")
    var = csv("results/robustness/qre_variants.csv")
    m = var[var.variant == "(main)"].iloc[0]
    lab = {"lam": ("$\\lambda$", "decision precision (1/win-prob.)"), "nu": ("$\\nu$", "weight on the objective's own value"),
           "kappa": ("$\\kappa$", "cost of early positioning"), "psi": ("$\\psi$", "cost per travel minute"),
           "omega_k": ("$\\omega_k$", "cost per player at the pit"), "omega_m": ("$\\omega_m$", "cost per player on 2nd front")}
    rows = []
    for k, (sym, desc) in lab.items():
        r = cib.loc[k]
        rows.append([sym, desc, f"{r.point:.3f}", ci(r.lo, r.hi, 3)])
    rows += [["$\\ell$", "log-lik. LB-QRE / saturated", f"{m.ll_qre:,.0f}", f"{m.ll_saturated:,.0f}"],
             ["$H_{winmax}$", "log-lik. loss of $\\nu=1$", f"{cib.loc['dll_winmax', 'boot_mean']:.1f}",
              ci(cib.loc["dll_winmax", "lo"], cib.loc["dll_winmax", "hi"], 1)]]
    write_table("tab_qre", ["", "Meaning", "Est.", "95\\% CI"], rows, align="llrr", caption=
                f"LB-QRE global parameters ({int(m.n_contests):,} contests, 19 teams with own parameters; 300 series-block bootstrap replicates).",
                label="tab:qre", midrules=(6,))


def t_shapley(N):
    cib = csv("results/robustness/qre_bootstrap_ci.csv").set_index("quantity")
    nt = csv("results/robustness/named_tests.csv")
    rows = []
    for k, nm in (("observed_gap", "observed gap"), ("gap_full", "model gap (all channels)"), ("gap_none", "model gap (channels off)"),
                  ("shap_L", "Shapley: Lanchester"), ("shap_C", "Shapley: curvature of $W$"), ("shap_F", "Shapley: second front"),
                  ("shap_B", "Shapley: behavioural")):
        r = cib.loc[k]
        rows.append([nm, f(r.point, 2, True, sign=True), ci(r.lo, r.hi, 2, True), f"{r.p_gt0:.2f}"])
    n0 = len(rows)
    names = ["$H_{Simpson}$ raw gap", "$H_{Simpson}$ matched (logit)", "$H_{trade}$ slope"]
    for nm, (_, r) in zip(names, nt.iterrows()):
        rows.append([nm + f" [{tex_escape(r.theory_direction).replace('>=', '$\\geq$').replace('<', '$<$')}]", f(r.estimate, 3, sign=True), ci(r.lo, r.hi, 3), f"$p$={r.p_two_sided:.3f}"])
    write_table("tab_shapley", ["Quantity", "Est.", "95\\% CI", "$P(>0)$"], rows, align="lrrr", caption=
                "Decomposition of the trailing-minus-leading commitment gap (pp) and the named tests (theory direction in brackets).",
                label="tab:shapley", midrules=(3, n0))
    for i, k in enumerate(("simpRaw", "simpMatched", "trade")):
        r = nt.iloc[i]
        N[k] = f(r.estimate, 3, sign=True)
        N[k + "CI"] = ci(r.lo, r.hi, 3)
        N[k + "P"] = f"{r.p_two_sided:.3f}"


def t_values(N):
    ev = csv("results/values/event_values.csv")
    col = "value_at_W=0.5 (points)"
    v = ev.groupby(["kind", "type"]).agg(val=(col, "first"), n=("n", "first"), a=("a", "first")).reset_index()
    order = ["HORDE", "RIFTHERALD", "DRAGON_1", "DRAGON_2", "DRAGON_3", "DRAGON_4", "ATAKHAN", "BARON_NASHOR", "ELDER",
             "FIGHT", "FIGHT_OBJ", "TOWER"]
    nm = {"HORDE": "Void grubs", "RIFTHERALD": "Rift Herald", "DRAGON_1": "Dragon 1", "DRAGON_2": "Dragon 2",
          "DRAGON_3": "Dragon 3 (soul point)", "DRAGON_4": "Dragon 4 (soul)", "ATAKHAN": "Atakhan", "BARON_NASHOR": "Baron",
          "ELDER": "Elder", "FIGHT": "Fight, per kill (away from objectives)", "FIGHT_OBJ": "Fight, per kill (near objective)", "TOWER": "Tower"}
    rows = []
    for t in order:
        w = v[(v.kind == "wpa") & (v.type == t)].iloc[0]
        r = v[(v.kind == "result") & (v.type == t)].iloc[0]
        rows.append([nm[t], f"{int(w.n):,}", f"{w.a:.2f}", f(w.val, 1), f(r.val, 1)])
    write_table("tab_values", ["Event", "$n$", "$a$", "WPA (180 s)", "Total effect"], rows, caption=
                "Event values in win-probability points at $W=0.5$, from outcome-only regressions on the archive "
                "($\\mathrm{logit}\\,W_{t+180} = a\\,\\mathrm{logit}\\,W_t + \\beta(t)\\,\\mathrm{dir}$); `total effect' replaces $W_{t+180}$ by the result.",
                label="tab:values")


def build(N):
    for fn in (t_data, t_tracker, t_rlayer, t_vlayer, t_selection, t_patterns, t_hasse_tagnet, t_fights, t_qre, t_shapley, t_values):
        fn(N)
    return "main tables"


if __name__ == "__main__":
    run_main(build)
