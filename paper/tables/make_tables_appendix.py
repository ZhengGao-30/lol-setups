"""Appendix tables (all numbers read from result files)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

OBJN = {"DRAGON": "Dragon", "HORDE": "Grubs", "RIFTHERALD": "Herald", "ATAKHAN": "Atakhan", "BARON_NASHOR": "Baron",
        "ELDER": "Elder"}


def t_vocab(N):
    v = csv("results/r_layer/vocabulary.csv")
    rows = [[OBJN.get(r.objective, r.objective), f"$s{int(r.offset_s):+d}$" if r.offset_s else "$s$", f"{r.n}", f"{r.distinct_partitions}",
             f"{r.distinct_shapes}", f"{r.rarefied_distinct_at_n_min:.1f}" if r.rarefied_distinct_at_n_min == r.rarefied_distinct_at_n_min else "--"]
            for _, r in v.iterrows()]
    write_table("tab_vocab", ["Objective", "Time", "$n$", "Partitions", "Shapes", "Rarefied"], rows, caption=
                "Formation vocabulary on the R layer (five alive, risk set). Rarefied = expected distinct partitions at the smallest $n$.",
                label="tab:vocab")


def t_qre_variants(N):
    var = csv("results/robustness/qre_variants.csv")
    lab = {"(main)": "main", "_hold": "drop held/interp.", "_law_v5": "V5 fight law", "_nocost": "no costs (original spec.)",
           "_nocost_law_v5": "no costs + V5 law", "_r1000": "pit radius 1,000", "_r2000": "pit radius 2,000",
           "_result": "`result' values", "_winmax": "$\\nu=1$ ($H_{winmax}$)"}
    rows = []
    for _, r in var.iterrows():
        rows.append([lab.get(r.variant, r.variant), f"{int(r.n_contests):,}", f"{r.ll_qre:,.0f}", f"{r.global_lam:.2f}", f"{r.global_nu:.3f}",
                     f(r.observed_gap, 3), f(r.model_gap, 3), f(r.shap_L, 4), f(r.shap_C, 4), f(r.shap_F, 4), f(r.shap_B, 4)])
    write_table("tab_qre_variants", ["Variant", "$n$", "$\\ell$", "$\\lambda$", "$\\nu$", "Obs. gap", "Model gap", "$\\phi_L$",
                                     "$\\phi_C$", "$\\phi_F$", "$\\phi_B$"], rows, star=True, size="\\scriptsize",
                caption="LB-QRE specification variants. All Shapley signs agree with the main fit in the six non-degenerate variants; the two no-cost variants collapse to $\\lambda=0$.", label="tab:qrevariants")


def t_team_params(N):
    tp = csv("results/robustness/qre_team_parameter_ci.csv")
    tc = csv("results/qre/team_parameters.csv").rename(columns={"Unnamed: 0": "team"}).set_index("team")
    teams = tc.drop(index="other", errors="ignore").sort_values("contests", ascending=False).index
    rows = []
    for t in teams:
        g = tp[tp.team == t].set_index("param")
        cell = lambda p, nd: f"{g.loc[p, 'point']:.{nd}f} [{g.loc[p, 'lo']:.{nd}f}, {g.loc[p, 'hi']:.{nd}f}]" if p in g.index else "--"
        rows.append([tex_escape(t), f"{int(tc.loc[t, 'contests'])}", cell("lam", 1), cell("kappa", 2), cell("psi", 2), cell("omega_k", 2), cell("omega_m", 2)])
    write_table("tab_team_params", ["Team", "Contests", "$\\lambda$", "$\\kappa$", "$\\psi$", "$\\omega_k$", "$\\omega_m$"], rows,
                star=True, size="\\scriptsize", align="lrlllll",
                caption="Team LB-QRE parameters with 95\\% series-block bootstrap intervals (shrunk towards the global values).",
                label="tab:teamparams")


def t_usage(N):
    u = csv("results/fingerprints_K6/usage.csv").sort_values("n_setups", ascending=False)
    d = csv("results/fingerprints_K6/distance_to_league.csv").rename(columns={"Unnamed: 0": "team"}).set_index("team")
    rows = [[tex_escape(r.team), f"{int(r.n_setups)}"] + [f"{r[f'occ_pattern{i}']:.2f}" for i in range(6)] +
            [f"{d.loc[r.team, 'tv_to_league_mean']:.3f}" if r.team in d.index else "--"] for _, r in u.iterrows()]
    write_table("tab_usage", ["Team", "Setups"] + [f"P{i}" for i in range(6)] + ["TV to league"], rows, size="\\scriptsize",
                caption="Pattern usage by team ($K=6$). Intervals in Fig.~\\ref{fig:fingerprints}; split-half reliability is low.",
                label="tab:usage")


def t_context(N):
    c = csv("results/context/coefficients_v_K6.csv")
    feats = list(dict.fromkeys(c.feature))
    rows = []
    for p in sorted(c.pattern.unique()):
        g = c[c.pattern == p].set_index("feature")
        rows.append([f"P{p}"] + [f"{g.loc[ft, 'coef_per_sd']:+.2f}{'$^*$' if g.loc[ft, 'sig'] else ''}" if ft in g.index else "--" for ft in feats])
    hdr = [""] + [tex_escape(x).replace("\\_", " ") for x in feats]
    write_table("tab_context", hdr, rows, star=True, size="\\scriptsize", caption=
                f"Soft-label multinomial logit of pattern occupancy (reference pattern {int(c.ref.iloc[0])}), coefficients per SD; "
                "$^*$ = 95\\% interval excludes 0 (500 series-block resamples).", label="tab:context")


def t_hasse_robust(N):
    rows = []
    for k, lab in (("hasse_v", "main (13 snapshots, radius 1,900)"), ("hasse_v7", "7 snapshots $s{-}90..s$"),
                   ("hasse_vr1500", "formation radius 1,500"), ("hasse_vr2500", "formation radius 2,500")):
        d = csv(f"results/{k}/selection.csv").set_index("K").sort_index()
        d = d.loc[[K for K in (3, 4, 5, 6) if K in d.index]]
        rows.append([lab] + [f"{d.loc[K, 'test_loglik_per_setup']:.2f}" for K in d.index] +
                    [f"{int(d.test_loglik_per_setup.idxmax())}", f"{int(d.bic.idxmin())}", f"{int(d.icl.idxmin())}"])
    write_table("tab_hasse_robust", ["Layer", "$K{=}3$", "4", "5", "6", "best held-out", "BIC", "ICL"], rows, star=True,
                caption="Discovery-layer robustness: held-out log-likelihood per setup for $K=3..6$ and the $K$ chosen by each criterion.",
                label="tab:hasserobust")


def t_wfit(N):
    rows = []
    for k, lab in (("pooled", "pooled, rule-specific objective terms (main)"), ("r2509", "same-rule games only"), ("all", "all games, shared terms")):
        ph = csv(f"results/winsurface/{k}/fit_by_phase.csv")
        cal = csv(f"results/winsurface/{k}/calibration_deciles.csv")
        rows.append([lab] + [f"{x:.3f}" for x in ph.logloss] + [f"{x:.3f}" for x in ph.brier] + [f(np.abs(cal.pred - cal.obs).max(), 1, pct=True)])
    write_table("tab_wfit", ["$W$ specification", "LL 1--13", "14--24", "25+", "Brier 1--13", "14--24", "25+", "max cal. dev. (pp)"],
                rows, star=True, size="\\scriptsize", caption="Out-of-fold fit of the outcome-only win surface $W$ by game phase.", label="tab:wfit")


def t_t1(N):
    a3 = csv("results/t1/A3_faker_vs_other_mids.csv")
    meas = {"in_pit": "in pit block at $s$", "rank": "arrival rank", "dpit": "geodesic distance to pit", "alone": "alone",
            "main": "in main block", "deg": "MST degree", "btw": "MST betweenness"}
    rows = [[meas[r.measure], f"{r.flagged_mean:.3f}" if r.measure != "dpit" else f"{r.flagged_mean:,.0f}",
             f"{r.others_mean:.3f}" if r.measure != "dpit" else f"{r.others_mean:,.0f}",
             f(r["diff"], 3 if r.measure != "dpit" else 0), ci(r.lo, r.hi, 3 if r.measure != "dpit" else 0)] for _, r in a3.iterrows()]
    write_table("tab_faker", ["Measure", "Faker", "Other mids", "Diff.", "95\\% CI"], rows, caption=
                f"A3: Faker against other starting mid laners in pre-spawn snapshots ($n$ = {int(a3.n_flagged.iloc[0])} vs {int(a3.n_others.iloc[0]):,}).",
                label="tab:faker")
    a5 = csv("results/t1/A5_state_behaviour.csv")
    rows = [[r.group, r.state, f"{r.n}", f"{r.commit_share:.3f}", f"{r.polarisation:.3f}", f"{r.mean_k:.2f}"] for _, r in a5.iterrows()]
    write_table("tab_t1_state", ["Group", "State", "$n$", "Commit", "Polar.", "Mean $k$"], rows, caption=
                "A5: T1 and all other teams by gold state at the commitment time (commit = $k\\geq3$; polarisation = $k\\in\\{0,5\\}$).",
                label="tab:t1state")
    led = csv("results/t1/A7_worlds2025_final_ledger.csv")
    games = {g: i + 1 for i, g in enumerate(dict.fromkeys(led.game))}
    on = {"DRAGON": "Drag", "ELDER": "Elder", "HORDE": "Grubs", "RIFTHERALD": "Herald", "ATAKHAN": "Atak", "BARON_NASHOR": "Baron"}
    rows = []
    for _, r in led.sort_values(["game", "spawn_s"], key=lambda s: s.map(games) if s.name == "game" else s).iterrows():
        tk = {100: "B", 200: "R"}.get(r.taken_by, "--") if r.taken_by == r.taken_by else "--"
        rows.append([f"{games[r.game]}", f"{on[r.objective]} {int(r.idx)}", f"{r.minute:.1f}", f"{r.W_blue:.2f}", f"{int(r.gold_diff_blue):+,}",
                     f"({int(r.blue_k)},{int(r.blue_m)},{int(r.blue_tau)})", f"({int(r.red_k)},{int(r.red_m)},{int(r.red_tau)})",
                     f"{r.p_hat_blue:.2f}", f"{r.p_star_blue_break_even:.2f}", f"{100 * r.value_obj_blue:+.1f}", tk])
    write_table("tab_ledger", ["G", "Spawn", "Min", "$W_B$", "GD$_B$", "Blue", "Red", "$\\hat p$", "$\\pi^*$", "$V_B$", "Took"], rows,
                align="rlrrrllrrrc", star=True, size="\\scriptsize",
                caption="A7: decision ledger of the Worlds 2025 final (T1 3--2 KT Rolster). Actions $(k,m,\\tau)$ at the commitment "
                "time, blue's fight probability $\\hat p$, break-even $\\pi^*$ and the objective's value to blue (pts).", label="tab:ledger")


def build(N):
    for fn in (t_vocab, t_qre_variants, t_team_params, t_usage, t_context, t_hasse_robust, t_wfit, t_t1):
        fn(N)
    return "appendix tables"


if __name__ == "__main__":
    run_main(build)
