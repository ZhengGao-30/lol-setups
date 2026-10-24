"""Pool the per-game tracker validation (results/tracker/*_eval.csv, *_formation.csv) into
results/tracker/summary.md and summary_*.csv: error quantiles, shares under 300/600/1000 units, by state,
per game, formation agreement, and error versus local crowding (other champions within 2000 units)."""
import glob

import numpy as np
import pandas as pd

from . import io

OUT = io.RESULTS / "tracker"
QS = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def describe(e):
    d = {f"q{int(q * 100)}": e.quantile(q) for q in QS}
    d.update({"n": len(e), "mean": e.mean(), "lt300": (e < 300).mean(), "lt600": (e < 600).mean(), "lt1000": (e < 1000).mean()})
    return pd.Series(d)


def main():
    ev = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "*_eval.csv")))], ignore_index=True)
    fo = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "*_formation.csv")))], ignore_index=True)
    ev = ev[ev["alive"]]
    ng = ev["name"].nunique()
    pooled = describe(ev["err"]).to_frame("pooled").T
    by_state = ev.groupby("state")["err"].apply(describe).unstack()
    by_state["share"] = ev["state"].value_counts(normalize=True)
    per_game = ev.groupby("name")["err"].apply(describe).unstack()
    # crowding: other champions (both teams) within 2000 units of the true position at the same minute
    cr = []
    for _, g in ev.groupby(["name", "minute"]):
        xy = g[["x", "y"]].to_numpy(float)
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        cr.append(pd.Series((d < 2000).sum(1) - 1, index=g.index))
    ev["crowd"] = pd.concat(cr)
    ev["crowd_bin"] = pd.cut(ev["crowd"], [-1, 0, 1, 2, 3, 9], labels=["0", "1", "2", "3", "4+"])
    by_crowd = ev.groupby("crowd_bin", observed=True)["err"].apply(describe).unstack()
    by_crowd["share"] = ev["crowd_bin"].value_counts(normalize=True)
    form = fo.groupby("n_alive").agg(n=("exact", "size"), exact=("exact", "mean"), rand=("rand", "mean"))
    form.loc["all"] = [len(fo), fo["exact"].mean(), fo["rand"].mean()]
    fgame = fo.groupby("name").agg(exact=("exact", "mean"), rand=("rand", "mean"))
    for n, d in (("pooled", pooled), ("by_state", by_state), ("per_game", per_game), ("by_crowding", by_crowd),
                 ("formation", form), ("formation_per_game", fgame)):
        d.to_csv(OUT / f"summary_{n}.csv")
    ev[["name", "minute", "pid", "state", "err", "crowd"]].to_parquet(OUT / "eval_pooled.parquet", index=False)
    def r(d):
        d = d.round(3).reset_index()
        rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
        rows += ["| " + " | ".join(map(str, x)) + " |" for x in d.itertuples(index=False)]
        return chr(10).join(rows)
    md = [f"# Tracker validation (odd V5 minutes, templates and pins from even minutes), {ng} games\n",
          "Position error in map units (the map is 14,820 wide; 1000 units is about 7% of it).\n",
          "## Pooled\n", r(pooled), "\n## By state\n", r(by_state), "\n## By local crowding (champions within 2000 units)\n",
          r(by_crowd), "\n## Formation agreement with V5 (exact partition, Rand index)\n", r(form),
          f"\n## Per game (median error: min {per_game['q50'].min():.0f}, median {per_game['q50'].median():.0f}, "
          f"max {per_game['q50'].max():.0f}; share <600: min {per_game['lt600'].min():.2f}, "
          f"median {per_game['lt600'].median():.2f})\n", "See summary_per_game.csv.\n"]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
