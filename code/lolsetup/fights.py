"""Local numbers at every kill and at every fight, from V5 kill events plus the 2-fps minimap tracks.

Deaths (stochastic Lanchester, test S7): for every CHAMPION_KILL, the players of each team present = alive
within FIGHT_R (geodesic) of the kill position at the last live track frame at least LOOKBACK_S before the
kill, united with the kill's V5 participants (victim, killer, assisters: present by definition); the
track-only counts are kept for sensitivity. Also each side's mean total gold per alive player (V5, interpolated). Under the stochastic square law the
next death falls on the outnumbered side with probability n_more / (n_more + n_fewer) (equal effectiveness):
the logistic slope of [victim on the larger side] on ln(n_more / n_fewer) is -1; under the linear law it is 0.

Fights (Lanchester -> Tullock contest function): kills are chained into a fight while each next kill is within
FIGHT_GAP_S of the previous one and within FIGHT_LINK_R of the fight's first kill. Numbers (tracked at
FIGHT_PRE_S before the first kill, united with all V5 participants of the fight) and gold per player; the winner is the team with more kills in the fight (ties are
kept with winner = None).

Usage:  python -m lolsetup.fights -> fits/deaths_numbers.parquet, fits/fights.parquet
"""
import numpy as np
import pandas as pd

from . import io
from .navgraph import NavGraph
from .run_clock import vod_name

FIGHT_R = 2000.0
LOOKBACK_S = 1.0
FIGHT_GAP_S, FIGHT_LINK_R, FIGHT_PRE_S = 15.0, 3000.0, 5.0
MATCH_S = 1.5


def _frame_before(gs, t):
    """Index of the last tracked frame at or before t (within MATCH_S), else -1."""
    j = int(np.searchsorted(gs, t, side="right")) - 1
    return j if j >= 0 and t - gs[j] <= MATCH_S else -1


def local_set(nav, X, Y, ST, j, x, y):
    """Participants (1..10) alive and within FIGHT_R (geodesic) of (x, y) at track frame j."""
    alive = ST[j] != "dead"
    ok = alive & np.isfinite(X[j])
    if not ok.any():
        return set()
    d = nav.d(nav.node(X[j][ok], Y[j][ok]), nav.node(x, y)[0])
    return {int(i) + 1 for i in np.flatnonzero(ok)[d <= FIGHT_R]}


def participants(e):
    return {e["victimId"], e["killerId"], *e.get("assistingParticipantIds", [])}


def counts(present):
    return sum(1 for i in present if i <= 5), sum(1 for i in present if i > 5)


def game_rows(nav, gid, tr):
    tl = io.load_v5(gid)
    frames = tl["frames"]
    t_ms = np.array([f["timestamp"] for f in frames], float)
    gold = np.array([[f["participantFrames"][str(i)]["totalGold"] for i in range(1, 11)] for f in frames], float)
    kills = [e for f in frames for e in f["events"] if e["type"] == "CHAMPION_KILL" and e.get("killerId", 0) > 0]
    tr = tr.sort_values(["k", "pid"])
    ks = tr["k"].unique()
    gs = tr.groupby("k")["game_s"].first().to_numpy()
    X = tr["x"].to_numpy().reshape(len(ks), 10)
    Y = tr["y"].to_numpy().reshape(len(ks), 10)
    ST = tr["state"].to_numpy().reshape(len(ks), 10)

    def gold_per_player(t, j):
        g = np.array([np.interp(1000 * t, t_ms, gold[:, i]) for i in range(10)])
        alive = ST[j] != "dead"
        return [float(g[s][alive[s]].mean()) if alive[s].any() else np.nan for s in (slice(0, 5), slice(5, 10))]

    deaths = []
    for e in kills:
        t = e["timestamp"] / 1000
        j = _frame_before(gs, t - LOOKBACK_S)
        if j < 0:
            continue
        tracked = local_set(nav, X, Y, ST, j, e["position"]["x"], e["position"]["y"])
        nb, nr = counts(tracked | participants(e))
        tb, trd = counts(tracked)
        gb, gr = gold_per_player(gs[j], j)
        deaths.append({"RiotPlatformGameId": gid, "t": t, "x": e["position"]["x"], "y": e["position"]["y"],
                       "victim_side": 100 if e["victimId"] <= 5 else 200, "n_blue": nb, "n_red": nr,
                       "n_blue_tracked": tb, "n_red_tracked": trd, "gold_pp_blue": gb, "gold_pp_red": gr})
    fights, cur = [], []
    for e in sorted(kills, key=lambda e: e["timestamp"]):
        if cur and (e["timestamp"] - cur[-1]["timestamp"] > 1000 * FIGHT_GAP_S or
                    np.hypot(e["position"]["x"] - cur[0]["position"]["x"],
                             e["position"]["y"] - cur[0]["position"]["y"]) > FIGHT_LINK_R):
            fights.append(cur)
            cur = []
        cur.append(e)
    if cur:
        fights.append(cur)
    frows = []
    for f in fights:
        t0 = f[0]["timestamp"] / 1000 - FIGHT_PRE_S
        j = _frame_before(gs, t0)
        if j < 0:
            continue
        tracked = local_set(nav, X, Y, ST, j, f[0]["position"]["x"], f[0]["position"]["y"])
        nb, nr = counts(tracked | set().union(*(participants(e) for e in f)))
        tb, trd = counts(tracked)
        gb, gr = gold_per_player(gs[j], j)
        kb = sum(e["victimId"] > 5 for e in f)  # kills by blue = red victims
        kr = len(f) - kb
        frows.append({"RiotPlatformGameId": gid, "t0": t0, "x": f[0]["position"]["x"], "y": f[0]["position"]["y"],
                      "n_kills": len(f), "kills_blue": kb, "kills_red": kr, "n_blue": nb, "n_red": nr,
                      "n_blue_tracked": tb, "n_red_tracked": trd,
                      "gold_pp_blue": gb, "gold_pp_red": gr,
                      "winner": 100 if kb > kr else 200 if kr > kb else None})
    return deaths, frows


def main():
    nav = NavGraph()
    man = pd.read_csv(io.MANIFESTS / "vod_manifest.csv")
    man = man[man["set"].str.startswith("core")]
    deaths, fights = [], []
    for r in man.itertuples():
        path = io.ROOT / "fits" / "tracks" / f"{vod_name(r.GameId)}.parquet"
        if path.exists():
            d, f = game_rows(nav, r.RiotPlatformGameId, pd.read_parquet(path))
            deaths += d
            fights += f
    d, f = pd.DataFrame(deaths), pd.DataFrame(fights)
    d.to_parquet(io.ROOT / "fits" / "deaths_numbers.parquet", index=False)
    f.to_parquet(io.ROOT / "fits" / "fights.parquet", index=False)
    print(f"deaths: {len(d)} from {d['RiotPlatformGameId'].nunique()} games; fights: {len(f)} "
          f"(decisive {f['winner'].notna().sum()})")


if __name__ == "__main__":
    main()
