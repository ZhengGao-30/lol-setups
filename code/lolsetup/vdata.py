"""Shared V-layer loader for the baselines and downstream stages: one row per team-setup (game x objective x
spawn index x side) with the pre-spawn snapshot tensor and the context columns (kept separate: context never
enters the pattern layer).

Snapshot tensor X: (n, T, 5, F) over roles T,J,M,B,S with F = 7 zone one-hot + alive + 5 role one-hot + x, y
(minimap units scaled to [0, 1]; dead roles: zero position, alive = 0, as TAGNet keeps dead players with zero
health). Euclidean positions P: (n, T, 5, 2). Observed mask M: (n, T) (a snapshot without a live track frame is
missing; a setup needs >= MIN_OBS of the T pre-spawn snapshots and missing ones are filled from the nearest
observed one, which is also how TAGNet's complete-case filter is relaxed for tracked video).
"""
import numpy as np
import pandas as pd

from . import geometry, io

PRE = tuple(range(-90, 1, 15))   # s-90 .. s: the 7 pre-spawn snapshots
KEY = ["RiotPlatformGameId", "objective", "idx", "side"]
CLASS = {"DRAGON": 0, "HORDE": 0, "RIFTHERALD": 0, "ATAKHAN": 1, "BARON_NASHOR": 2, "ELDER": 2}
MAP = 14820.0
MIN_OBS = 6


def load(path=None, offsets=PRE, min_obs=MIN_OBS):
    d = pd.read_parquet(path or io.ROOT / "fits" / "v_layer_setups.parquet")
    games = io.core_games()[["GameId", "MatchId"]]
    d = d.merge(games, on="GameId", how="left")
    zi = {z: i for i, z in enumerate(geometry.ZONES)}
    T = len(offsets)
    X, P, M, ctx = [], [], [], []
    for key, g in d.groupby(KEY, sort=True):
        first = g.iloc[0]
        g = g.set_index("offset_s").reindex(list(offsets))
        obs = g["partition"].notna().to_numpy()
        if obs.sum() < min_obs:
            continue
        idx = np.where(obs)[0]
        near = idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(1)]
        x = np.zeros((T, 5, 7 + 1 + 5 + 2)); p = np.zeros((T, 5, 2))
        for t in range(T):
            r = g.iloc[near[t]]
            for k, ch in enumerate(io.ROLE_SHORT):
                al = bool(r[f"alive_{ch}"])
                x[t, k, 8 + k] = 1
                if al:
                    x[t, k, zi[r[f"zone_{ch}"]]] = 1
                    x[t, k, 7] = 1
                    p[t, k] = (r[f"x_{ch}"] / MAP, r[f"y_{ch}"] / MAP)
        x[..., -2:] = p
        last = g.iloc[near[-1]]
        X.append(x); P.append(p); M.append(obs)
        ctx.append({"RiotPlatformGameId": key[0], "objective": key[1], "idx": key[2], "side": key[3],
                    "cls": CLASS[key[1]], "MatchId": str(first["MatchId"]), "team": first["team"],
                    "opponent": first["opponent"], "event": first["event"], "spawn_s": first["spawn_s"],
                    "gold_diff": last["gold_diff"], "obj_up_end": bool(last["obj_up"]),
                    "taken_by_team": first["taken_by_team"], "n_obs": int(obs.sum()),
                    "q_hold": float(np.nanmean(g["q_hold"].to_numpy(float))) if g["q_hold"].notna().any() else np.nan})
    return np.stack(X).astype(np.float32), np.stack(P).astype(np.float32), np.stack(M), pd.DataFrame(ctx)
