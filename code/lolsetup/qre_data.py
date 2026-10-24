"""Data for the LB-QRE layer: per objective spawn, the state values and each team's observed action.

State: the out-of-fold W_blue at the last V5 minute frame at least values.PRE_S before the spawn, its game minute and
both teams' drake counts. Values for blue (win-probability points) come from values.py (outcome-only archive
regressions; primary: win probability added over values.HORIZON_S; W evaluated right after an event misses what
the event later turns into): obj_self / obj_opp = value of the objective gained by blue / by red (DRAGON resolved to the gainer's next
drake index, capped at 4); fight_win / fight_lose = value of a fight won / lost by FIGHT_MARGIN kills; front_win /
front_lose = value of one tower. Red's values are the antisymmetric flip (qre.Values.flip).
Actions (V layer, risk set): both teams are read at the commitment time t_c of the spawn = first at-risk snapshot in
[0, +180] s where some team has >= 3 players at the pit at R = PIT_R, else the last at-risk snapshot (objective taken
next, or window end: censored); m = alive players in zone 'far' at t_c; tau = 1 if the team already had >= k players
at the pit at the spawn (k > 0).
"""
import pickle

import numpy as np
import pandas as pd

from . import geometry, io, qre, values

FIGHT_MARGIN = 2.0
REACH_S = 60.0     # feasibility: alive players whose geodesic distance to the pit at the spawn is <= REACH_S * MOVE_SPEED
MOVE_SPEED = 381.0  # median movement speed (units / s)


def load_event_models(kind="wpa"):
    """kind: 'wpa' (primary, win probability added over values.HORIZON_S) or 'result' (sensitivity)."""
    with open(io.ROOT / "fits" / "event_values.pkl", "rb") as f:
        return pickle.load(f)[kind]


def state_values(models, W, minute, objective, drakes_b=0, drakes_r=0, fight_margin=FIGHT_MARGIN):
    """qre.Values for blue at a state with out-of-fold win probability W at game minute `minute`."""
    def v(typ, gainer, mag=1.0):
        return float(values.value(models, typ, W, minute, gainer, mag)[0])
    if objective == "DRAGON":
        t_b, t_r = f"DRAGON_{min(drakes_b + 1, 4)}", f"DRAGON_{min(drakes_r + 1, 4)}"
    else:
        t_b = t_r = objective
    return qre.Values(obj_self=v(t_b, 100), obj_opp=v(t_r, 200),
                      fight_win=v("FIGHT", 100, fight_margin), fight_lose=v("FIGHT", 200, fight_margin),
                      front_win=v("TOWER", 100), front_lose=v("TOWER", 200))


def observed_actions(v, R=geometry.PIT_R, window=180, commit_k=3):
    """One row per (spawn, side) with (n, k, m, tau) read at the commitment time t_c of the spawn: the first at-risk
    snapshot in [0, window] where some team has >= commit_k players at the pit, else the last at-risk snapshot
    (the objective is taken right after it, or the window ends: censored). Both teams are read at the same t_c."""
    v = v[(v["offset_s"] >= 0) & (v["offset_s"] <= window) & v["obj_up"]].copy()
    v["k"] = sum(((v[f"dpit_{r}"] <= R) & v[f"alive_{r}"]).astype(int) for r in io.ROLE_SHORT)
    v["m"] = sum(((v[f"zone_{r}"] == "far") & v[f"alive_{r}"]).astype(int) for r in io.ROLE_SHORT)
    unc = v["q_hold"] + (v["q_interp"] if "q_interp" in v else 0.0)
    v["q_uncertain"] = unc
    rows = []
    for key, g in v.groupby(["RiotPlatformGameId", "objective", "idx"]):
        both = g.groupby("offset_s").filter(lambda x: x["side"].nunique() == 2)
        if both.empty:
            continue
        kmax = both.groupby("offset_s")["k"].max()
        hit = kmax[kmax >= commit_k]
        t_c = float(hit.index[0]) if len(hit) else float(kmax.index[-1])
        for side, d in both[both["offset_s"] == t_c].groupby("side"):
            d = d.iloc[0]
            at0 = both[(both["offset_s"] == 0) & (both["side"] == side)]
            k0 = int(at0["k"].iloc[0]) if len(at0) else 0
            eta = np.sort([float(at0[f"dpit_{r}"].iloc[0]) / MOVE_SPEED for r in io.ROLE_SHORT if at0[f"alive_{r}"].iloc[0]])                 if len(at0) else np.zeros(0)
            travel = np.r_[0.0, np.cumsum(eta)] / 60.0               # minutes of travel of the k nearest players
            travel = np.r_[travel, np.full(6 - len(travel), travel[-1] if len(travel) else 0.0)]
            kf = int(sum(((at0[f"dpit_{r}"] <= REACH_S * MOVE_SPEED) & at0[f"alive_{r}"]).astype(int).iloc[0]
                         for r in io.ROLE_SHORT)) if len(at0) else int(d["alive"])
            rows.append({"RiotPlatformGameId": key[0], "objective": key[1], "idx": key[2], "side": side,
                         "team": d["team"], "spawn_s": d["spawn_s"], "decision_offset": t_c, "committed": bool(len(hit)),
                         "n": int(d["alive"]), "k": int(d["k"]), "m": int(min(d["m"], d["alive"] - d["k"])),
                         "tau": int(d["k"] > 0 and k0 >= d["k"]), "kf": max(kf, int(d["k"])), "travel": travel, "q_hold": d["q_hold"], "q_uncertain": d["q_uncertain"]})
    return pd.DataFrame(rows)


def build(v_setups, w_oof, frames_state, models, oc, R=geometry.PIT_R, hold_max=None, costs=True):
    """Contests ready for qre.npl: list of (game, (team_i, team_j)), choices, team names, per-contest info."""
    acts = observed_actions(v_setups, R)
    if hold_max is not None:                      # robustness: drop decisions read from mostly held (covered-icon) frames
        acts = acts[acts["q_uncertain"].fillna(0) <= hold_max]
    travel_by = {(r.RiotPlatformGameId, r.objective, r.idx, r.side): r.travel for r in acts.itertuples()}
    pairs = acts.drop(columns="travel").pivot_table(index=["RiotPlatformGameId", "objective", "idx"], columns="side",
                             values=["team", "n", "k", "m", "tau", "spawn_s", "kf"], aggfunc="first").dropna()
    teams = sorted(set(acts["team"]))
    tix = {t: i for i, t in enumerate(teams)}
    w_by = {g: d.sort_values("frame_ms") for g, d in w_oof.groupby("RiotPlatformGameId")}
    fs_by = {g: d.sort_values("frame_ms") for g, d in frames_state.groupby("RiotPlatformGameId")}
    games, choices, info = [], [], []
    for (gid, obj, idx), r in pairs.iterrows():
        if gid not in w_by or obj not in ("DRAGON", "ELDER", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR"):
            continue
        t_pre = r[("spawn_s", 100)] - values.PRE_S
        wg, fg = w_by[gid], fs_by[gid]
        j = int(np.searchsorted(wg["frame_ms"].to_numpy() / 1000, t_pre, side="right")) - 1
        jf = int(np.searchsorted(fg["frame_ms"].to_numpy() / 1000, t_pre, side="right")) - 1
        if j < 0 or jf < 0:
            continue
        W, minute = float(wg["W_blue_oof"].iloc[j]), float(r[("spawn_s", 100)] / 60)
        st = fg.iloc[jf]
        vals = state_values(models, W, minute, obj, int(st["b_drakes"]), int(st["r_drakes"]))
        n_b, n_r = int(r[("n", 100)]), int(r[("n", 200)])
        if n_b == 0 or n_r == 0:
            continue
        lng = float(np.log(st["b_gold"] / st["r_gold"]))
        kf_b, kf_r = int(r[("kf", 100)]), int(r[("kf", 200)])
        g, A_b, A_r = qre.build_game(n_b, n_r, vals, lng, oc, kf_b, kf_r, travel_by[(gid, obj, idx, 100)] if costs else None,
                                     travel_by[(gid, obj, idx, 200)] if costs else None)
        a_b = tuple(int(r[(c, 100)]) for c in ("k", "m", "tau"))
        a_r = tuple(int(r[(c, 200)]) for c in ("k", "m", "tau"))
        games.append((g, (tix[r[("team", 100)]], tix[r[("team", 200)]])))
        choices.append((int(np.flatnonzero((A_b == a_b).all(1))[0]), int(np.flatnonzero((A_r == a_r).all(1))[0])))
        info.append({"RiotPlatformGameId": gid, "objective": obj, "idx": idx, "W_blue": W, "minute": minute,
                     "gd": float(st["b_gold"] - st["r_gold"]), "lng": lng, "n_b": n_b, "n_r": n_r, "kf_b": kf_b, "kf_r": kf_r, "travel_b": travel_by[(gid, obj, idx, 100)], "travel_r": travel_by[(gid, obj, idx, 200)],
                     "drakes_b": int(st["b_drakes"]), "drakes_r": int(st["r_drakes"]),
                     **{f"v_{k}": x for k, x in vals.__dict__.items()}})
    return games, choices, teams, pd.DataFrame(info)
