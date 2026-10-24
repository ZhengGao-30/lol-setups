"""Table I numbers: games, series, teams, spawns, team-setups and position quality of the V layer.

    python -m lolsetup.data_stats   -> results/data/*.csv, results/data/summary.md
"""
import pandas as pd

from . import io

OUT = io.RESULTS / "data"
KEY = ["RiotPlatformGameId", "objective", "idx", "side"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(io.ROOT / "fits" / "v_layer_setups.parquet")
    g = io.core_games()
    sp = io.spawns()
    sp = sp[sp["RiotPlatformGameId"].isin(g["RiotPlatformGameId"])]
    ev = g.assign(event=g["OverviewPage"]).groupby("event").agg(games=("GameId", "size"), series=("MatchId", "nunique"))
    ev["teams"] = [len(set(g.loc[g["OverviewPage"] == e, "Blue"]) | set(g.loc[g["OverviewPage"] == e, "Red"])) for e in ev.index]
    ev["spawns"] = [int(sp["RiotPlatformGameId"].isin(g.loc[g["OverviewPage"] == e, "RiotPlatformGameId"]).sum()) for e in ev.index]
    ev.loc["total"] = [ev["games"].sum(), ev["series"].sum(), len(set(g["Blue"]) | set(g["Red"])), ev["spawns"].sum()]
    ev.to_csv(OUT / "table1_events.csv")
    by_obj = sp.groupby("objective").agg(spawns=("idx", "size"), taken=("kill_s", lambda s: int(s.notna().sum())),
                                         median_delay_s=("kill_s", lambda s: float("nan")))
    sp["delay"] = sp["kill_s"] - sp["spawn_s"]
    by_obj["median_delay_s"] = sp.groupby("objective")["delay"].median()
    setups = v.groupby(KEY).ngroups
    by_obj["team_setups"] = v.groupby(KEY).size().reset_index().groupby("objective").size()
    by_obj.to_csv(OUT / "table1_objectives.csv")
    q = v.groupby("objective")[["q_det", "q_interp", "q_pin", "q_hold"]].mean()
    q.to_csv(OUT / "position_sources_by_objective.csv")
    teams = pd.concat([g[["Blue"]].rename(columns={"Blue": "team"}), g[["Red"]].rename(columns={"Red": "team"})]).value_counts("team").rename("games")
    teams.to_csv(OUT / "games_per_team.csv")
    md = lambda d: d.round(3).reset_index().to_markdown(index=False) if hasattr(d, "to_markdown") else d.round(3).to_string()
    lines = ["# Data statistics (V layer)", "", f"Games {len(g)}, team-setups {setups}, team-snapshots {len(v)}.", "",
             ev.to_string(), "", by_obj.round(2).to_string(), "", q.round(3).to_string(), "", teams.to_string()]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
