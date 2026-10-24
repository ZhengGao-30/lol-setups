"""Fit LB-QRE on the V layer and decompose the trailing-leading commitment gap into four channels.

Steps:
  1. contests from qre_data.build (state values: values.py total effects; fight law: analysis_fights, Tullock
     gamma_n, gamma_g on tracked + V5 participant counts; delta_tau = 0 and gamma_front = 1 unless estimated)
  2. NPL estimates of (lam, nu, kappa) per team (teams with < MIN_TEAM contests share a pooled 'other' team)
  3. fit check: QRE log-likelihood vs a saturated multinomial benchmark (empirical action shares by objective
     group x gold state x n alive), as an over-identification check; prediction of the trailing-leading gap
  4. Shapley decomposition of gap = P(k >= 3 | trailing) - P(k >= 3 | leading) over the channels
       L  Lanchester: numbers decide fights (off: gamma_n = 0)
       C  curvature: stakes depend on the state (off: values evaluated at W = 0.5 at the same minute)
       F  second front (off: front values = 0)
       B  team-specific behaviour (off: every team uses the pooled parameters)
     each of the 2^4 subsets is re-solved at the observed states; Shapley values allocate the gap to channels.
Costs (this version): besides the early-positioning cost kappa the action carries a travel-time cost psi (minutes of travel of
the k players nearest to the pit at the spawn) and linear per-player costs omega_k (pit) and omega_m (second front);
without them the fit collapses to lam = 0 (see results/qre_nocost). The fight law defaults to tracked counts only.
Gold state: GD before the spawn / archive SD at that minute (as in analysis_v).

Usage:  python -m lolsetup.run_qre -> fits/qre_fit.pkl, results/qre/*.csv + summary.md
"""
import itertools
import math
import pickle

import numpy as np
import pandas as pd

from . import io, qre, qre_data
from .analysis_r import gold_sd_by_minute, md

MIN_TEAM = 20
PARAMS = ["lam", "nu", "kappa", "psi", "omega_k", "omega_m"]  # decision precision, objective valuation, positioning / travel / per-player costs
CHANNELS = ("L", "C", "F", "B")


LAWS = {"v5": "tracked + V5 participants", "tracked": "tracked only"}


def outcome_params(law="tracked"):
    f = pd.read_csv(io.RESULTS / "fights" / "fight_laws.csv")
    f = f[(f["model"] == "Tullock fight") & (f["counts"] == LAWS[law])].set_index("param")["est"]
    return qre.Outcome(gamma_n=float(f["gamma_n"]), gamma_g=float(f["gamma_g"]), delta_tau=0.0, gamma_front=1.0)


def states(info, sd):
    a = info["gd"] / info["minute"].round().astype(int).clip(upper=sd.index.max()).map(sd)
    return pd.cut(a, [-np.inf, -0.5, 0.5, np.inf], labels=["trailing", "even", "leading"]).astype(str)


def commit_share(si, n_actions_k):
    return float(si[n_actions_k >= 3].sum())


def solve_all(games_spec, theta, channels, oc, models, pooled):
    """Equilibrium P(k >= 3) for both teams of every contest under the switched-on channels."""
    out = []
    oc_x = qre.Outcome(oc.gamma_n if "L" in channels else 0.0, oc.gamma_g, oc.delta_tau, oc.gamma_front)
    for spec in games_spec:
        vals = spec["vals"] if "C" in channels else spec["vals_flat"]
        if "F" not in channels:
            vals = qre.Values(vals.obj_self, vals.obj_opp, vals.fight_win, vals.fight_lose, 0.0, 0.0)
        g, A_b, A_r = qre.build_game(spec["n_b"], spec["n_r"], vals, spec["lng"], oc_x, spec["kb"], spec["kr"], spec["tr_b"], spec["tr_r"])
        tb, tr = (theta[spec["tb"]], theta[spec["tr"]]) if "B" in channels else (pooled, pooled)
        sb, sr, _ = qre.solve(g, tuple(tb), tuple(tr))
        out.append((commit_share(sb, A_b[:, 0]), commit_share(sr, A_r[:, 0])))
    return np.array(out)


def gap(p, st_b):
    """P(k >= 3 | trailing) - P(k >= 3 | leading) over team-contests (blue state st_b; red's is mirrored)."""
    st_r = np.where(st_b == "trailing", "leading", np.where(st_b == "leading", "trailing", "even"))
    pk = np.r_[p[:, 0], p[:, 1]]
    st = np.r_[st_b, st_r]
    return float(pk[st == "trailing"].mean() - pk[st == "leading"].mean())


def shapley(vals):
    """vals: dict frozenset(channels) -> quantity. Returns channel -> Shapley value."""
    n = len(CHANNELS)
    out = {}
    for c in CHANNELS:
        tot = 0.0
        others = [x for x in CHANNELS if x != c]
        for r in range(n):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += w * (vals[frozenset(S) | {c}] - vals[frozenset(S)])
        out[c] = tot
    return out


def prepare(kind="wpa", R=None, hold_max=None, law="tracked", costs=True):
    """Contests, choices, team remap and the per-contest solver specs (everything the fit and the bootstrap reuse)."""
    from . import geometry
    R = geometry.PIT_R if R is None else R
    v = pd.read_parquet(io.ROOT / "fits" / "v_layer_setups.parquet")
    w = pd.read_parquet(io.ROOT / "fits" / "winsurface_oof_pooled.parquet")
    fs = pd.read_parquet(io.ROOT / "fits" / "frames_state.parquet")
    models = qre_data.load_event_models(kind)
    oc = outcome_params(law)
    games, choices, teams, info = qre_data.build(v, w, fs, models, oc, R=R, hold_max=hold_max, costs=costs)
    counts = pd.Series([t for _, pair in games for t in pair]).value_counts()
    small = {i for i in range(len(teams)) if counts.get(i, 0) < MIN_TEAM}
    remap = {i: (len(teams) if i in small else i) for i in range(len(teams))}
    games = [(g, (remap[a], remap[b])) for g, (a, b) in games]
    names = teams + ["other"]
    sd = gold_sd_by_minute()
    st_b = states(info, sd).to_numpy()
    spec = []
    for (g, (tb, tr)), (_, r) in zip(games, info.iterrows()):
        vals = qre.Values(r.v_obj_self, r.v_obj_opp, r.v_fight_win, r.v_fight_lose, r.v_front_win, r.v_front_lose)
        flat = qre_data.state_values(models, 0.5, r.minute, r.objective, r.drakes_b, r.drakes_r)
        spec.append({"vals": vals, "vals_flat": flat, "n_b": int(r.n_b), "n_r": int(r.n_r), "kb": int(r.kf_b), "kr": int(r.kf_r), "tr_b": r.travel_b if costs else None, "tr_r": r.travel_r if costs else None, "tb": tb, "tr": tr,
                     "lng": float(r.lng)})
    mid = io.core_games().set_index("RiotPlatformGameId")["MatchId"].astype(str)
    info["MatchId"] = info["RiotPlatformGameId"].map(mid).to_numpy()
    return {"games": games, "choices": choices, "teams": teams, "names": names, "info": info, "counts": counts,
            "small": small, "st_b": st_b, "spec": spec, "oc": oc, "models": models}


def fit(P, idx=None, fix_nu=False, decompose=True):
    """NPL fit (+ fit check vs saturated shares + 16-subset Shapley decomposition) on contests `idx` (default all;
    repeated indices = bootstrap resample)."""
    idx = np.arange(len(P["games"])) if idx is None else np.asarray(idx)
    games = [P["games"][i] for i in idx]
    choices = [P["choices"][i] for i in idx]
    spec = [P["spec"][i] for i in idx]
    st_b = P["st_b"][idx]
    info = P["info"].iloc[idx]
    theta, strat, hist = qre.npl(games, choices, None, len(P["names"]), fix_nu=fix_nu)
    pooled_theta, _, _ = qre.npl([(g, (0, 0)) for g, _ in games], choices, None, 1, fix_nu=fix_nu)
    ll_qre = float(sum(np.log(si[a]) + np.log(sj[b]) for (si, sj), (a, b) in zip(strat, choices)))
    st_r = np.where(st_b == "trailing", "leading", np.where(st_b == "leading", "trailing", "even"))
    obs = pd.DataFrame({"cell": [f"{o}|{s_}|{x['n_b']}|{x['kb']}" for o, s_, x in zip(info["objective"], st_b, spec)],
                        "a": [a for a, _ in choices]})
    obs_r = pd.DataFrame({"cell": [f"{o}|{s_}|{x['n_r']}|{x['kr']}" for o, s_, x in zip(info["objective"], st_r, spec)],
                          "a": [b for _, b in choices]})
    both = pd.concat([obs, obs_r])
    shares = both.groupby(["cell", "a"]).size() / both.groupby("cell").size()
    ll_sat = float(np.log(shares.loc[list(zip(both["cell"], both["a"]))]).sum())
    res = {"theta": theta, "pooled_theta": pooled_theta[0], "npl_history": hist, "ll_qre": ll_qre, "ll_saturated": ll_sat,
           "n_contests": len(games), "converged": bool(hist[-1][1] < 1e-5), "n_iter": len(hist)}
    obs_k = np.array([[qre.actions(s["n_b"], s["kb"])[a, 0] >= 3, qre.actions(s["n_r"], s["kr"])[b, 0] >= 3]
                      for s, (a, b) in zip(spec, choices)], float)
    res["observed_gap"] = gap(obs_k, st_b)
    if decompose:
        gaps = {}
        for r_ in range(len(CHANNELS) + 1):
            for S in itertools.combinations(CHANNELS, r_):
                p = solve_all(spec, theta, set(S), P["oc"], P["models"], pooled_theta[0])
                gaps[frozenset(S)] = gap(p, st_b)
        res["gaps"] = {"".join(sorted(k)): v for k, v in gaps.items()}
        res["shapley"] = shapley(gaps)
    return res


def write_summary(P, res, out, tag):
    oc, names, counts, small = P["oc"], P["names"], P["counts"], P["small"]
    teams = P["teams"]
    th = pd.DataFrame(res["theta"], columns=PARAMS[:res["theta"].shape[1]], index=names)
    th["lam_obj"], th["lam_rest"] = th["lam"] * th["nu"], th["lam"]
    th["contests"] = [counts.get(i, 0) for i in range(len(teams))] + [sum(counts.get(i, 0) for i in small)]
    th.to_csv(out / "team_parameters.csv")
    gaps, sh = res["gaps"], res["shapley"]
    full, none = gaps["".join(sorted(CHANNELS))], gaps[""]
    lines = [f"# LB-QRE (V layer){tag}", "", f"Contests: {res['n_contests']}; teams with own parameters: {len(teams) - len(small)}; "
             f"fight law: gamma_n = {oc.gamma_n:.2f}, gamma_g = {oc.gamma_g:.2f}.", "",
             f"NPL: {res['n_iter']} iterations, converged = {res['converged']}; global (lam, nu, kappa, psi, omega_k, omega_m) = {np.round(res['pooled_theta'], 3).tolist()}.", "",
             f"Log-likelihood: QRE {res['ll_qre']:.1f}  vs saturated cell shares {res['ll_saturated']:.1f}.", "",
             f"Commitment gap P(k>=3 | trailing) - P(k>=3 | leading): observed {res['observed_gap']:+.3f}; "
             f"full model {full:+.3f}; all channels off {none:+.3f}.", "",
             "Shapley decomposition of the model gap:", "", md(pd.Series(sh, name="shapley").to_frame()), "",
             "Team parameters:", "", md(th.round(3))]
    (out / "summary.md").write_text(chr(10).join(lines), encoding="utf-8")
    print(chr(10).join(lines))


def main(kind="wpa", R=None, hold_max=None, tag="", fix_nu=False, law="tracked", costs=True):
    out = io.RESULTS / f"qre{tag}"
    out.mkdir(parents=True, exist_ok=True)
    P = prepare(kind, R, hold_max, law, costs)
    res = fit(P, fix_nu=fix_nu)
    res.update({"teams": P["names"], "outcome": P["oc"].__dict__, "kind": kind, "R": R, "law": law, "costs": costs, "hold_max": hold_max, "fix_nu": fix_nu})
    with open(io.ROOT / "fits" / f"qre_fit{tag}.pkl", "wb") as f:
        pickle.dump(res, f)
    write_summary(P, res, out, tag)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="wpa", choices=["wpa", "result"])
    ap.add_argument("--R", type=float, default=None, help="pit radius for the observed actions (geodesic units)")
    ap.add_argument("--hold-max", type=float, default=None, help="drop decisions with a held-share above this")
    ap.add_argument("--fix-nu", action="store_true", help="H_winmax: nu = 1")
    ap.add_argument("--law", default="tracked", choices=list(LAWS), help="fight law: tracked + V5 participant counts, or tracked only")
    ap.add_argument("--no-cost", action="store_true", help="original specification: only the early-positioning penalty")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.kind, a.R, a.hold_max, a.tag, a.fix_nu, a.law, not a.no_cost)
