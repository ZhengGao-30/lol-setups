"""Fig. 15: the arrival race. Share of spawns in which the first team to put three players at the pit
(R = 1500) takes the objective within 180 s, by objective group; and the time to that first three-player arrival."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

GROUPS = ["DRAGON", "DRAGON_2+", "HORDE", "RIFTHERALD", "ATAKHAN", "BARON_NASHOR", "BARON_2+", "ELDER"]


def build(N):
    ar = csv("results/v_layer/arrival_race.csv")
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 1.85), gridspec_kw=dict(wspace=0.55, width_ratios=[1.2, 1]))
    a = axs[0]
    rows = []
    for g in GROUPS:
        d = ar[ar.group == g]
        dec = d[d.race.isin(["blue", "red"])]
        took = (dec.race == dec.outcome).mean() if len(dec) else np.nan
        rows.append((g, len(dec), took, (dec.outcome == "neither").mean() if len(dec) else np.nan))
    rows = pd.DataFrame(rows, columns=["group", "n", "took", "neither"])
    y = np.arange(len(rows))
    a.barh(y, rows.took, color=C["blue"])
    for i, r in rows.iterrows():
        a.text(0.02, i, f"n={r.n}", va="center", fontsize=6.0, color="white")
    a.set_yticks(y, [OBJ_LABEL[g] for g in rows.group], fontsize=6.5)
    a.invert_yaxis()
    a.set_xlim(0, 1.0)
    a.axvline(0.5, color=C["grey"], lw=0.6, ls=":")
    a.set_xlabel("race winner takes it")
    a.set_title("(a) first to three || race winner takes the objective")
    b = axs[1]
    t = np.minimum(ar.first3_blue, ar.first3_red)
    t = t[np.isfinite(t)]
    b.hist(t, bins=np.arange(-120, 195, 15), color=C["orange"])
    b.axvline(0, color="k", lw=0.5, ls=":")
    b.set_xlabel("first 3-at-pit (s from spawn)")
    b.set_ylabel("spawns")
    b.set_title("(b) timing || first three-player arrival")
    r = rows.set_index("group")
    for g, k in (("ATAKHAN", "Atakhan"), ("BARON_NASHOR", "Baron"), ("DRAGON", "DragonOne"), ("DRAGON_2+", "DragonTwo"),
                 ("RIFTHERALD", "Herald")):
        N[f"race{k}"] = f"{r.loc[g, 'took']:.2f}"
        N[f"race{k}N"] = f"{int(r.loc[g, 'n'])}"
    N["raceMedianT"] = f"{np.median(t):.0f}"
    return save(fig, "fig15_arrival")


if __name__ == "__main__":
    run_main(build)
