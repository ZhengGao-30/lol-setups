"""Outcome-only values of objectives, fights and towers (archive), two ways.

W (winsurface.py) conditions on the current gold, towers and buffs, so evaluating it right after an objective
leaves out everything the objective later turns into (gold, towers, the next objective); read that way, a baron
comes out slightly negative. Decision payoffs need what the event turns into. For every event type e:
  wpa (primary)  win probability added over HORIZON_S, the standard sports-analytics value: least squares on
                     logit W_blue(x_post) = a_e * logit W_blue(x_pre) + beta_e(t) * s
                 with x_post the last frame at or before event + HORIZON_S (the result if the game ended earlier).
                 Uses states only, so it is not inflated by who the teams are.
  result         total effect on the eventual result (sensitivity; inflated when the gaining team is stronger in
                 ways W misses):
                     logit P(blue wins) = a_e * logit W_blue(x_pre) + beta_e(t) * s
with x_pre the last minute frame at least PRE_S before the event (out-of-fold W, never in-sample), s = +1 if
blue gained the event and -1 if red did (fights: the signed kill margin of the fight), and beta_e(t) a quadratic
B-spline in game minutes. No intercept, so the model is antisymmetric. Values are evaluated inside each event
type's observed time range (2.5-97.5% of its events; minutes outside are clamped to it) because the shared spline
basis is unidentified where an event never happens (e.g. Elder before ~26 minutes). Value for blue of gaining e at state x:
    Lambda(a_e logit W + beta_e(t)) - W   (and of red gaining it: Lambda(a_e logit W - beta_e(t)) - W).
Event types: DRAGON (by index 1..4; soul point and soul included), ELDER, HORDE (camp, owner = >= 2 of 3 grubs),
RIFTHERALD, ATAKHAN, BARON_NASHOR, TOWER, FIGHT (V5 kill clusters as in fights.py, more than OBJ_NEAR_S from any
objective kill: their value does not already contain the objective, which the QRE payoff pays separately) and
FIGHT_OBJ (the fights near objective kills, reported only). Objectives whose rules
changed (HORDE, RIFTHERALD, ATAKHAN, BARON_NASHOR) are fitted on same-rule games (e2025b) only; the rest on all
eras. Selection caveat: the team that takes an event can be stronger in ways W(x_pre) misses; this is a total
effect conditional on the pre-event state, not a randomised one.

Usage:  python -m lolsetup.values -> fits/event_values.pkl, results/values/*.csv
"""
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from . import io, winsurface

PRE_S = 30.0
HORIZON_S = 180.0
RIDGE = 1e-3  # wpa: ridge on the time spline, per observation
FIGHT_GAP_S, FIGHT_LINK_R = 15.0, 3000.0
OBJ_NEAR_S = 60.0  # fights this close to an objective kill: FIGHT_OBJ
GRUB_CAMP_S = 90.0
REGIME = ("HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR")


def events(gid):
    """(type, t_s, gainer side, magnitude) for one game from its V5 timeline."""
    fr = io.load_v5(gid)["frames"]
    ev = [e for f in fr for e in f["events"]]
    out, drakes, grubs = [], {100: 0, 200: 0}, []
    for e in ev:
        t = e["timestamp"] / 1000
        if e["type"] == "ELITE_MONSTER_KILL" and e.get("killerTeamId") in (100, 200):
            side, mt = e["killerTeamId"], e["monsterType"]
            if mt == "DRAGON":
                if e.get("monsterSubType") == "ELDER_DRAGON":
                    out.append(("ELDER", t, side, 1.0))
                else:
                    drakes[side] += 1
                    out.append((f"DRAGON_{min(drakes[side], 4)}", t, side, 1.0))
            elif mt == "HORDE":
                grubs.append((t, side))
            elif mt in ("RIFTHERALD", "ATAKHAN", "BARON_NASHOR"):
                out.append((mt, t, side, 1.0))
        elif e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
            out.append(("TOWER", t, 300 - e["teamId"], 1.0))  # teamId = the tower's owner
    camp = []
    for t, side in grubs + [(np.inf, None)]:
        if camp and t - camp[0][0] > GRUB_CAMP_S:
            n_b = sum(s == 100 for _, s in camp)
            if 2 * n_b != len(camp):
                out.append(("HORDE", camp[0][0], 100 if 2 * n_b > len(camp) else 200, 1.0))
            camp = []
        if side is not None:
            camp.append((t, side))
    obj_times = [t for typ, t, _, _ in out if typ != "TOWER"] + [t for t, _ in grubs]
    kills = sorted((e for e in ev if e["type"] == "CHAMPION_KILL"), key=lambda e: e["timestamp"])
    cur = []
    for e in kills + [None]:
        if cur and (e is None or e["timestamp"] - cur[-1]["timestamp"] > 1000 * FIGHT_GAP_S or
                    np.hypot(e["position"]["x"] - cur[0]["position"]["x"],
                             e["position"]["y"] - cur[0]["position"]["y"]) > FIGHT_LINK_R):
            kb = sum(k["victimId"] > 5 for k in cur)
            margin = kb - (len(cur) - kb)
            if margin != 0:
                t0 = cur[0]["timestamp"] / 1000
                near = any(abs(t0 - te) <= OBJ_NEAR_S for te in obj_times)
                out.append(("FIGHT_OBJ" if near else "FIGHT", t0, 100 if margin > 0 else 200, float(abs(margin))))
            cur = []
        if e is not None:
            cur.append(e)
    return out


def event_table():
    pg = io.played_games()
    pg = pg.assign(era=winsurface.era_of(pg))
    w = pd.read_parquet(io.ROOT / "fits" / "winsurface_oof_pooled.parquet")
    fs = pd.read_parquet(io.ROOT / "fits" / "frames_state.parquet")[["RiotPlatformGameId", "blue_win"]].drop_duplicates()
    rows = []
    for gid in sorted(set(w["RiotPlatformGameId"])):
        try:
            evs = events(gid)
        except FileNotFoundError:
            continue
        wg = w[w["RiotPlatformGameId"] == gid].sort_values("frame_ms")
        tms = wg["frame_ms"].to_numpy() / 1000
        for typ, t, side, mag in evs:
            j = int(np.searchsorted(tms, t - PRE_S, side="right")) - 1
            if j < 0:
                continue
            jp = int(np.searchsorted(tms, t + HORIZON_S, side="right")) - 1
            ended = t + HORIZON_S > tms[-1] + 60  # past the last minute frame: the game ended within the horizon
            rows.append({"RiotPlatformGameId": gid, "type": typ, "t": t, "gainer": side, "magnitude": mag,
                         "W_pre": float(wg["W_blue_oof"].iloc[j]), "minute_pre": float(wg["minute"].iloc[j]),
                         "W_post": np.nan if ended else float(wg["W_blue_oof"].iloc[jp]), "ended": ended})
    ev = pd.DataFrame(rows).merge(fs, on="RiotPlatformGameId").merge(pg[["RiotPlatformGameId", "MatchId", "era"]],
                                                                     on="RiotPlatformGameId")
    ev.loc[ev["ended"], "W_post"] = ev.loc[ev["ended"], "blue_win"].astype(float)
    ev["s"] = np.where(ev["gainer"] == 100, 1.0, -1.0) * ev["magnitude"]
    return ev


def _design(minute, lw, s):
    T = winsurface.time_basis(np.asarray(minute) + 0.0)
    return np.column_stack([lw, T * np.asarray(s)[:, None]])


def fit(ev, target="wpa"):
    """Per event type: coefficients (a, spline) of the antisymmetric model (no intercept)."""
    models = {}
    for typ, g in ev.groupby("type"):
        if typ in REGIME:
            g = g[g["era"] == "e2025b"]
        if len(g) < 200:
            continue
        lw = logit(np.clip(g["W_pre"].to_numpy(), 1e-4, 1 - 1e-4))
        X = _design(g["t"] / 60, lw, g["s"])
        if target == "wpa":
            y = logit(np.clip(g["W_post"].to_numpy(), 0.01, 0.99))  # a finished game counts as 1% / 99%
            P = RIDGE * len(g) * np.diag(np.r_[0.0, np.ones(X.shape[1] - 1)])  # ridge on the spline only
            coef = np.linalg.solve(X.T @ X + P, X.T @ y)
        else:
            coef = LogisticRegression(C=10.0, fit_intercept=False, max_iter=5000).fit(X, g["blue_win"]).coef_[0]
        lo, hi = np.percentile(g["t"] / 60, [2.5, 97.5])
        models[typ] = {"coef": coef, "n": len(g), "games": g["RiotPlatformGameId"].nunique(), "t_lo": lo, "t_hi": hi}
    return models


def value(models, typ, W_blue, minute, gainer=100, magnitude=1.0):
    """Change in blue's win probability when `gainer` gains event `typ` at state W_blue (scalar or arrays):
    Lambda(a logit W + beta(t) s) - Lambda(a logit W), zero when nothing happens."""
    c = models[typ]["coef"]
    lw = logit(np.clip(np.asarray(W_blue, float), 1e-4, 1 - 1e-4))
    minute = np.clip(np.atleast_1d(minute) + 0.0, models[typ]["t_lo"], models[typ]["t_hi"])  # stay in the support
    T = winsurface.time_basis(minute)
    beta = T @ c[1:]
    s = (1.0 if gainer == 100 else -1.0) * magnitude
    return expit(c[0] * lw + beta * s) - expit(c[0] * lw)


def main():
    out = io.RESULTS / "values"
    out.mkdir(parents=True, exist_ok=True)
    ev = event_table()
    ev.to_parquet(io.ROOT / "fits" / "event_table.parquet", index=False)
    both = {kind: fit(ev, kind) for kind in ("wpa", "result")}
    with open(io.ROOT / "fits" / "event_values.pkl", "wb") as f:
        pickle.dump(both, f)
    rows = []
    for kind, models in both.items():
        for typ in models:
            for minute in (5, 8, 12, 15, 20, 25, 30, 35):
                rows.append({"kind": kind, "type": typ, "minute": minute, "n": models[typ]["n"],
                             "value_at_W=0.5 (points)": 100 * float(value(models, typ, 0.5, minute)[0]),
                             "a": models[typ]["coef"][0]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "event_values.csv", index=False)
    for kind in both:
        print(f"== {kind} (value of gaining the event at W = 0.5, points)")
        print(tab[tab["kind"] == kind].pivot(index="minute", columns="type", values="value_at_W=0.5 (points)")
              .round(2).to_string())
        print({t: (both[kind][t]["n"], round(both[kind][t]["coef"][0], 3)) for t in both[kind]})


if __name__ == "__main__":
    main()
