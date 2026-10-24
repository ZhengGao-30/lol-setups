"""Robustness, bootstrap and named tests for the LB-QRE layer (all on the V layer).

    python -m lolsetup.run_robust boot --batch 3 --reps 5            # series-block bootstrap batch -> fits/qre_boot/
    python -m lolsetup.run_robust boot_summary                       # CIs over all batches -> results/robustness/
    python -m lolsetup.run_robust variants                           # table of qre_fit_<tag>.pkl variants
    python -m lolsetup.run_robust named                              # H_Simpson, H_trade (empirical, series bootstrap)

A bootstrap replicate resamples series (MatchId) with replacement, repeats the NPL fit and the 16-subset Shapley
decomposition on the resample (team set fixed), and also refits with nu = 1 (H_winmax) to record the log-likelihood
loss of the restriction. Paired contrasts are taken within replicates.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import io, qre, run_qre

OUT = io.RESULTS / "robustness"
BOOT = io.ROOT / "fits" / "qre_boot"


def cmd_boot(batch, reps, kind="wpa", R=None, hold_max=None, tag="", winmax=True, law="tracked"):
    BOOT.mkdir(parents=True, exist_ok=True)
    P = run_qre.prepare(kind, R, hold_max, law)
    series = P["info"]["MatchId"].to_numpy()
    uniq = np.unique(series)
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    rng = np.random.default_rng(1000 + batch)
    rows = []
    for r in range(reps):
        take = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_by[s] for s in take])
        res = run_qre.fit(P, idx)
        row = {"batch": batch, "rep": r, "n_contests": len(idx), "ll_qre": res["ll_qre"], "observed_gap": res["observed_gap"],
               "lam": res["pooled_theta"][0], "nu": res["pooled_theta"][1], "kappa": res["pooled_theta"][2], "psi": res["pooled_theta"][3] if len(res["pooled_theta"]) > 3 else np.nan, "omega_k": res["pooled_theta"][4] if len(res["pooled_theta"]) > 4 else np.nan, "omega_m": res["pooled_theta"][5] if len(res["pooled_theta"]) > 5 else np.nan,
               "global_lam_nu": float(res["theta"][:, 0].mean()), "converged": res["converged"],
               **{f"shap_{k}": v for k, v in res["shapley"].items()}, "gap_full": res["gaps"]["BCFL"], "gap_none": res["gaps"][""],
               "theta": res["theta"]}
        if winmax:
            fx = run_qre.fit(P, idx, fix_nu=True, decompose=False)
            row.update({"ll_nu1": fx["ll_qre"], "dll_winmax": res["ll_qre"] - fx["ll_qre"]})
        rows.append(row)
        with open(BOOT / f"{tag or 'main'}_b{batch}.pkl", "wb") as f:
            pickle.dump(rows, f)
        print(f"batch {batch} rep {r}: shapley {[round(res['shapley'][k], 3) for k in 'LCFB']} gap {res['gaps']['BCFL']:+.3f}", flush=True)


def cmd_boot_summary(tag="main"):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for f in sorted(BOOT.glob(f"{tag}_b*.pkl")) for r in pickle.load(open(f, "rb"))]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta"} for r in rows])
    full = pickle.load(open(io.ROOT / "fits" / "qre_fit.pkl", "rb"))
    out = []
    for c in [c for c in df.columns if c.startswith("shap_")] + ["gap_full", "gap_none", "observed_gap", "lam", "nu", "kappa", "psi", "omega_k", "omega_m", "dll_winmax"]:
        v = df[c].dropna()
        pt = full["shapley"].get(c[5:]) if c.startswith("shap_") else full["gaps"]["BCFL"] if c == "gap_full" else full["gaps"][""] \
            if c == "gap_none" else full["observed_gap"] if c == "observed_gap" else dict(zip(run_qre.PARAMS, full["pooled_theta"])).get(c)
        out.append({"quantity": c, "point": pt, "boot_mean": v.mean(), "lo": v.quantile(0.025), "hi": v.quantile(0.975),
                    "p_gt0": (v > 0).mean(), "n_reps": len(v)})
    pd.DataFrame(out).to_csv(OUT / "qre_bootstrap_ci.csv", index=False)
    df.to_csv(OUT / "qre_bootstrap_reps.csv", index=False)
    # paired contrasts between channels, within replicates
    pc = []
    for a, b in (("L", "C"), ("L", "F"), ("L", "B"), ("C", "F"), ("C", "B"), ("F", "B")):
        d = df[f"shap_{a}"] - df[f"shap_{b}"]
        pc.append({"contrast": f"{a}-{b}", "mean": d.mean(), "lo": d.quantile(0.025), "hi": d.quantile(0.975), "p_gt0": (d > 0).mean()})
    pd.DataFrame(pc).to_csv(OUT / "qre_shapley_paired_contrasts.csv", index=False)
    # team parameter intervals (lam_obj = lam * nu, kappa) across replicates
    th = np.stack([r["theta"] for r in rows])                 # reps x teams x 3
    names = full["teams"]
    tp = []
    for i, n in enumerate(names):
        for j, p in enumerate(run_qre.PARAMS[:th.shape[2]]):
            v = th[:, i, j]
            tp.append({"team": n, "param": p, "point": full["theta"][i, j], "lo": np.quantile(v, 0.025), "hi": np.quantile(v, 0.975)})
        v = th[:, i, 0] * th[:, i, 1]
        tp.append({"team": n, "param": "lam_obj", "point": full["theta"][i, 0] * full["theta"][i, 1],
                   "lo": np.quantile(v, 0.025), "hi": np.quantile(v, 0.975)})
    pd.DataFrame(tp).to_csv(OUT / "qre_team_parameter_ci.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string())


def cmd_variants():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted((io.ROOT / "fits").glob("qre_fit*.pkl")):
        r = pickle.load(open(f, "rb"))
        tag = f.stem.replace("qre_fit", "") or "(main)"
        rows.append({"variant": tag, "kind": r.get("kind", "wpa"), "R": r.get("R"), "hold_max": r.get("hold_max"),
                     "fix_nu": r.get("fix_nu", False), "n_contests": r["n_contests"], "ll_qre": r["ll_qre"], "ll_saturated": r["ll_saturated"],
                     "global_lam": r["pooled_theta"][0], "global_nu": r["pooled_theta"][1], "global_kappa": r["pooled_theta"][2], "global_psi": r["pooled_theta"][3] if len(r["pooled_theta"]) > 3 else np.nan,
                     "observed_gap": r["observed_gap"], "model_gap": r["gaps"]["BCFL"], **{f"shap_{k}": v for k, v in r["shapley"].items()}})
    df = pd.DataFrame(rows)
    main_ = df[df["variant"] == "(main)"]
    if len(main_):
        for k in "LCFB":   # sign agreement of every Shapley value with the main fit
            df[f"sign_agree_{k}"] = np.sign(df[f"shap_{k}"]) == np.sign(main_[f"shap_{k}"].iloc[0])
    df.to_csv(OUT / "qre_variants.csv", index=False)
    print(df.round(3).to_string())


def team_contests(P):
    """One row per team-contest: committed (k >= 3), non-commit second-front share, state, matched covariates."""
    rows = []
    oc = P["oc"]
    for i, (s, (a, b)) in enumerate(zip(P["spec"], P["choices"])):
        v = s["vals"]
        st = P["st_b"][i]
        info = P["info"].iloc[i]
        for side, n_own, n_opp, act, state, stake, lg in (
                ("blue", s["n_b"], s["n_r"], qre.actions(s["n_b"], s["kb"])[a], st, v.fight_win - v.fight_lose, s["lng"]),
                ("red", s["n_r"], s["n_b"], qre.actions(s["n_r"], s["kr"])[b],
                 {"trailing": "leading", "leading": "trailing"}.get(st, "even"), v.fight_lose - v.fight_win, -s["lng"])):
            p_hat = float(expit(oc.gamma_n * np.log(n_own / n_opp) + oc.gamma_g * lg))
            rows.append({"MatchId": info["MatchId"], "team": P["names"][s["tb"] if side == "blue" else s["tr"]], "objective": info["objective"], "state": state, "k": int(act[0]), "m": int(act[1]),
                         "commit": int(act[0] >= 3), "p_hat": p_hat, "logit_p": float(logit(np.clip(p_hat, 1e-3, 1 - 1e-3))),
                         "stake": abs(stake), "n_own": n_own})
    return pd.DataFrame(rows)


def _logit_coef(d, y, cols):
    import statsmodels.api as sm
    X = sm.add_constant(d[cols].astype(float))
    return sm.Logit(d[y], X).fit(disp=0, method="bfgs", maxiter=200).params


def cmd_named(boot=1000, kind="wpa"):
    OUT.mkdir(parents=True, exist_ok=True)
    P = run_qre.prepare(kind)
    d = team_contests(P)
    d.to_csv(OUT / "team_contests.csv", index=False)
    d["trailing"] = (d["state"] == "trailing").astype(float)
    d["leading"] = (d["state"] == "leading").astype(float)
    d["stake_z"] = (d["stake"] - d["stake"].mean()) / d["stake"].std()
    d["nc_trade"] = ((d["commit"] == 0) & (d["n_own"] >= 2)).astype(int)
    series = d["MatchId"].to_numpy()
    uniq = np.unique(series)
    idx_by = {s: np.flatnonzero(series == s) for s in uniq}
    rng = np.random.default_rng(7)

    def stats(x):
        raw = x.loc[x["state"] == "trailing", "commit"].mean() - x.loc[x["state"] == "leading", "commit"].mean()
        sub = x[x["state"] != "even"]
        c = _logit_coef(x, "commit", ["trailing", "leading", "logit_p", "stake_z"])
        nc = x[x["commit"] == 0].copy()
        nc["trade"] = (nc["m"] >= 2).astype(int)
        t = _logit_coef(nc, "trade", ["logit_p"])
        return {"raw_gap_trailing_minus_leading": raw, "matched_logit_trailing_minus_leading": c["trailing"] - c["leading"],
                "trade_slope_on_logit_p": t["logit_p"], "n_noncommit": len(nc), "n_trailing": int((x["state"] == "trailing").sum()),
                "n_leading": int((x["state"] == "leading").sum())}
    pt = stats(d)
    reps = []
    for b in range(boot):
        take = rng.choice(uniq, len(uniq))
        reps.append(stats(d.iloc[np.concatenate([idx_by[s] for s in take])]))
    reps = pd.DataFrame(reps)
    rows = []
    for k in ("raw_gap_trailing_minus_leading", "matched_logit_trailing_minus_leading", "trade_slope_on_logit_p"):
        rows.append({"test": {"raw_gap_trailing_minus_leading": "H_Simpson (raw: trailing - leading commit share)",
                              "matched_logit_trailing_minus_leading": "H_Simpson (matched on p_hat and stake: logit coefficient)",
                              "trade_slope_on_logit_p": "H_trade (slope of 2+ on the second front among non-committers, per logit p_hat)"}[k],
                     "estimate": pt[k], "lo": reps[k].quantile(0.025), "hi": reps[k].quantile(0.975),
                     "p_two_sided": min(1.0, 2 * min((reps[k] > 0).mean(), (reps[k] < 0).mean())),
                     "theory_direction": {"raw_gap_trailing_minus_leading": "< 0 (trailing contest less)",
                                          "matched_logit_trailing_minus_leading": ">= 0", "trade_slope_on_logit_p": "< 0"}[k]})
    pd.DataFrame(rows).to_csv(OUT / "named_tests.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(), "\n", {k: pt[k] for k in ("n_noncommit", "n_trailing", "n_leading")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["boot", "boot_summary", "variants", "named"])
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kind", default="wpa")
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--hold-max", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-winmax", action="store_true")
    a = ap.parse_args()
    if a.cmd == "boot":
        cmd_boot(a.batch, a.reps, a.kind, a.R, a.hold_max, a.tag, not a.no_winmax)
    elif a.cmd == "boot_summary":
        cmd_boot_summary(a.tag or "main")
    elif a.cmd == "variants":
        cmd_variants()
    else:
        cmd_named(kind=a.kind)
