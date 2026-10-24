"""Outcome-only, zero-sum win-probability surface W (explanation layer, L4).

    logit W_blue(x, t) = sum_k beta_k(t) * [phi_k(x_blue) - phi_k(x_red)] + beta_side(t)
    W_red = 1 - W_blue  (antisymmetric by construction: difference features + side term only)

beta(t) varies with game time through a B-spline basis in minutes (interaction design). The gold
difference enters through a piecewise-linear (hinge) basis so curvature in gold can be read off.
Fitted by L2-penalised logistic regression; the penalty is chosen by series-grouped 5-fold CV and
predictions are cross-fitted (out-of-fold) so that downstream payoffs never use in-sample fits.
W never sees any setup choice. Values derived from W (e.g. dragon value) are 'state values under W',
not causal effects.

Eras: 'r2509' (primary) fits only games under the core set's objective rules (2025, patch >= 25.09), because
objective counts mean different things across rule regimes (e.g. 6 vs 3 grubs); 'all' uses the whole archive
as a reference fit.

Usage: python -m lolsetup.winsurface --era r2509
    -> fits/winsurface_<era>.pkl, fits/winsurface_oof_<era>.parquet, results/winsurface/<era>/*.csv
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from . import io

GOLD_KNOTS = (-8000, -4000, -2000, 0, 2000, 4000, 8000)
TIME_KNOTS = (0, 10, 20, 30, 40, 60)
DIFF_FEATURES = ("xp", "kills", "towers", "inhibs", "plates", "drakes", "barons", "grubs", "heralds", "atakhans",
                 "soul", "soul_point", "baron_buff", "elder_buff", "alive")
CS = np.logspace(-3, 1, 9)
# Objective features whose rules changed across seasons get regime-specific coefficients in the 'pooled' fit
REGIME_FEATURES = ("barons", "grubs", "heralds", "atakhans", "baron_buff")
ERAS = ("e2023", "e2024", "e2025a", "e2025b", "e2026")  # e2025a: 2025 before 25.09; e2025b: patch >= 25.09


def era_of(games):
    pn = games["Patch"].map(io.patch_num)
    y = games["Year"]
    return np.select([y == 2023, y == 2024, (y == 2025) & (pn < 25.09), (y == 2025) & (pn >= 25.09), y == 2026],
                     list(ERAS), default="other")


def time_basis(minute, degree=2):
    knots = np.r_[[TIME_KNOTS[0]] * degree, TIME_KNOTS, [TIME_KNOTS[-1]] * degree]
    n = len(knots) - degree - 1
    m = np.clip(np.asarray(minute, float), TIME_KNOTS[0], TIME_KNOTS[-1] - 1e-9)
    return np.column_stack([BSpline(knots, np.eye(n)[i], degree, extrapolate=False)(m) for i in range(n)])


def gold_basis(gd):
    gd = np.asarray(gd, float)
    cols = [gd / 1000.0] + [np.maximum(gd - k, 0) / 1000.0 for k in GOLD_KNOTS]
    return np.column_stack(cols)


def design(df, pooled=False):
    """Rows from blue's perspective. Swapping teams negates every column except the side block.
    pooled=True: REGIME_FEATURES enter as (difference x era indicator), i.e. one coefficient per rule regime."""
    T = time_basis(df["minute"])
    G = gold_basis(df["b_gold"] - df["r_gold"]) - gold_basis(df["r_gold"] - df["b_gold"])
    G = G / 2.0  # odd part only: f(gd) - f(-gd) keeps antisymmetry under team swap
    feats = [k for k in DIFF_FEATURES if not (pooled and k in REGIME_FEATURES)]
    D = np.column_stack([df[f"b_{k}"] - df[f"r_{k}"] for k in feats]).astype(float)
    D[:, feats.index("xp")] /= 1000.0  # positive rescaling only: keeps antisymmetry, helps lbfgs
    if pooled:
        R = np.column_stack([df[f"b_{k}"] - df[f"r_{k}"] for k in REGIME_FEATURES]).astype(float)
        D = np.hstack([D] + [R * (df["era"].to_numpy() == e)[:, None] for e in ERAS])
    X = [T[:, [j]] * G for j in range(T.shape[1])] + [T[:, [j]] * D for j in range(T.shape[1])]
    side = T  # beta_side(t) * s_blue with s_blue = +1
    return np.hstack(X + [side])


def fit(df, C, pooled=False):
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=5000)
    m.fit(design(df, pooled), df["blue_win"])
    return m


def main(era="r2509"):
    out = io.RESULTS / "winsurface" / era
    out.mkdir(parents=True, exist_ok=True)
    pooled = era == "pooled"
    df = pd.read_parquet(io.ROOT / "fits" / "frames_state.parquet")
    pg = io.played_games()
    df = df.merge(pd.DataFrame({"RiotPlatformGameId": pg["RiotPlatformGameId"], "era": era_of(pg)}),
                  on="RiotPlatformGameId", how="left")
    if era == "r2509":
        df = df[df["era"] == "e2025b"]
    df = df[df["minute"] >= 1].reset_index(drop=True)
    groups = df["MatchId"].fillna(df["RiotPlatformGameId"])
    gkf = GroupKFold(n_splits=5)
    scores = []
    for C in CS:
        oof = np.zeros(len(df))
        for tr, te in gkf.split(df, groups=groups):
            oof[te] = fit(df.iloc[tr], C, pooled).predict_proba(design(df.iloc[te], pooled))[:, 1]
        scores.append((C, log_loss(df["blue_win"], oof), brier_score_loss(df["blue_win"], oof)))
    sc = pd.DataFrame(scores, columns=["C", "logloss", "brier"])
    sc.to_csv(out / "cv_penalty.csv", index=False)
    best = sc.loc[sc["logloss"].idxmin(), "C"]
    oof = np.zeros(len(df))
    for tr, te in gkf.split(df, groups=groups):
        oof[te] = fit(df.iloc[tr], best, pooled).predict_proba(design(df.iloc[te], pooled))[:, 1]
    df["W_blue_oof"] = oof
    df[["RiotPlatformGameId", "frame_ms", "minute", "W_blue_oof"]].to_parquet(io.ROOT / "fits" / f"winsurface_oof_{era}.parquet")
    model = fit(df, best, pooled)
    with open(io.ROOT / "fits" / f"winsurface_{era}.pkl", "wb") as f:
        pickle.dump({"model": model, "C": best, "features": DIFF_FEATURES, "gold_knots": GOLD_KNOTS,
                     "time_knots": TIME_KNOTS}, f)
    # calibration by decile and by game phase (out-of-fold)
    df["bin"] = pd.qcut(df["W_blue_oof"], 10, duplicates="drop")
    cal = df.groupby("bin", observed=True).agg(pred=("W_blue_oof", "mean"), obs=("blue_win", "mean"), n=("blue_win", "size"))
    cal.to_csv(out / "calibration_deciles.csv")
    df["phase"] = pd.cut(df["minute"], [0, 14, 25, 100], labels=["01-13", "14-24", "25+"])
    ph = df.groupby("phase", observed=True).apply(
        lambda d: pd.Series({"logloss": log_loss(d["blue_win"], d["W_blue_oof"], labels=[0, 1]),
                             "brier": brier_score_loss(d["blue_win"], d["W_blue_oof"]), "n": len(d)}))
    ph.to_csv(out / "fit_by_phase.csv")
    # antisymmetry self-check: swap teams in the design and confirm W_red = 1 - W_blue
    sw = df.head(2000).copy()
    for k in ("gold",) + DIFF_FEATURES:
        sw[f"b_{k}"], sw[f"r_{k}"] = df.head(2000)[f"r_{k}"].values, df.head(2000)[f"b_{k}"].values
    Xs = design(sw, pooled)
    side_cols = time_basis(sw["minute"]).shape[1]
    Xs[:, -side_cols:] *= -1  # red perspective: s = -1
    w_red = model.predict_proba(Xs)[:, 1]
    w_blue = model.predict_proba(design(df.head(2000), pooled))[:, 1]
    err = float(np.max(np.abs(w_red - (1 - w_blue))))
    print(sc.to_string(index=False))
    print(f"best C={best}; frames={len(df)}; games={df['RiotPlatformGameId'].nunique()}; antisymmetry max error={err:.2e}")
    core = df[df["era"] == "e2025b"]  # the population the setups live in: compare every fit here
    print(f"on same-rule (e2025b) games: logloss={log_loss(core['blue_win'], core['W_blue_oof'], labels=[0, 1]):.4f} "
          f"brier={brier_score_loss(core['blue_win'], core['W_blue_oof']):.4f} frames={len(core)}")
    print(cal.to_string())
    print(ph.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", choices=["pooled", "r2509", "all"], default="pooled")
    main(ap.parse_args().era)
