"""Appendix (T1 / Faker) figures: A3 Faker's position measures vs other starting mids, A5/A8 T1 behaviour by
gold state and event, A7 the Worlds 2025 final ledger (every spawn, fight probability vs break-even)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

MEAS = {"in_pit": "in pit block at s", "rank": "arrival rank (1–5)", "dpit": "distance to pit / 1000",
        "alone": "alone (singleton)", "main": "in main block", "deg": "MST degree", "btw": "MST betweenness"}


def build(N):
    out = []
    a3 = csv("results/t1/A3_faker_vs_other_mids.csv")
    a5 = csv("results/t1/A5_state_behaviour.csv")
    a5c = csv("results/t1/A5_trailing_minus_leading_contrast.csv")
    a6 = csv("results/t1/A6_fight_shift.csv").iloc[0]
    a8 = csv("results/t1/A8_t1_by_event.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.4), gridspec_kw=dict(wspace=0.75, width_ratios=[1.3, 1, 0.8]))
    a = axs[0]
    y = np.arange(len(a3))
    sc = np.where(a3.measure == "dpit", 1e-3, 1.0)
    a.errorbar(a3["diff"] * sc, y, xerr=[(a3["diff"] - a3.lo) * sc, (a3.hi - a3["diff"]) * sc], fmt="o", ms=3, color=C["red"])
    a.axvline(0, color="k", lw=0.5)
    a.set_yticks(y, [MEAS[m] for m in a3.measure], fontsize=6.5)
    a.invert_yaxis()
    a.set_xlabel("Faker $-$ other mids")
    a.set_title("(a) A3 Faker || minus other mids")
    b = axs[1]
    for g, col in (("T1", C["red"]), ("others", C["grey"])):
        d = a5[a5.group == g].set_index("state").reindex(["trailing", "even", "leading"])
        b.plot(range(3), d.commit_share, "o-", color=col, label=g, ms=3)
    b.set_xticks(range(3), ["trailing", "even", "leading"])
    b.set_ylabel("commitment share ($k\\geq3$)")
    b.legend(fontsize=6.3)
    b.set_title("(b) A5 gold state || commitment share")
    c = axs[2]
    c.bar(a8.event, a8.commit_share, color=[C["sky"], C["blue"]])
    c.set_ylabel("T1 commitment share")
    c.set_title("(c) A8 events || T1 only")
    c.tick_params(axis="x", labelsize=6.3)
    out.append(save(fig, "figA_t1_faker"))
    a3i = a3.set_index("measure")
    N["fakerN"] = f"{int(a3i.n_flagged.iloc[0])}"
    N["fakerOthersN"] = f"{int(a3i.n_others.iloc[0]):,}"
    N["fakerDpit"] = f(a3i.loc["dpit", "diff"], 0)
    N["fakerDpitCI"] = ci(a3i.loc["dpit", "lo"], a3i.loc["dpit", "hi"], 0)
    N["fakerRank"] = f(a3i.loc["rank", "diff"], 2)
    N["fakerRankCI"] = ci(a3i.loc["rank", "lo"], a3i.loc["rank", "hi"], 2)
    c5 = a5c.set_index("contrast")
    N["tOneTL"] = f(c5.estimate.iloc[0], 3)
    N["tOneTLCI"] = ci(c5.lo.iloc[0], c5.hi.iloc[0], 3)
    N["othersTL"] = f(c5.estimate.iloc[1], 3)
    N["tOneMinusOthers"] = f(c5.estimate.iloc[2], 3)
    N["tOneMinusOthersCI"] = ci(c5.lo.iloc[2], c5.hi.iloc[2], 3)
    N["tOneFightRes"] = f(a6.mean_residual, 3)
    N["tOneFightResCI"] = ci(a6.lo, a6.hi, 3)
    N["tOneFights"] = f"{int(a6.n_fights_with_T1)}"
    e = a8.set_index("event")
    N["tOneMSI"] = f"{e.loc['MSI 2025', 'commit_share']:.3f}"
    N["tOneWorlds"] = f"{e.loc['Worlds 2025', 'commit_share']:.3f}"

    led = csv("results/t1/A7_worlds2025_final_ledger.csv")
    games = list(dict.fromkeys(led.game))
    fig, axs = plt.subplots(1, len(games), figsize=(TEXT_W, 2.5), sharey=True, gridspec_kw=dict(wspace=0.1))
    mk = {"DRAGON": "o", "ELDER": "D", "HORDE": "v", "RIFTHERALD": "^", "ATAKHAN": "s", "BARON_NASHOR": "*"}
    for i, (a, gm) in enumerate(zip(axs, games)):
        d = led[led.game == gm]
        for _, r in d.iterrows():
            col = SIDE_C.get(r.taken_by, C["grey"]) if r.taken_by == r.taken_by else C["grey"]
            a.scatter(r.minute, r.p_hat_blue, marker=mk.get(r.objective, "o"), s=14, color=col, edgecolors="k", linewidths=0.3, zorder=3)
            a.plot([r.minute, r.minute], [r.p_star_blue_break_even, r.p_hat_blue], color=C["grey"], lw=0.5)
            a.scatter(r.minute, r.p_star_blue_break_even, marker="_", s=14, color="k")
        a.set_title(f"game {i + 1}")
        a.set_xlabel("minute")
        a.set_ylim(0, 1)
    axs[0].set_ylabel("blue $\\hat p$ (marker) vs $\\pi^*$ (bar)")
    out.append(save(fig, "figA_final_ledger"))
    N["ledgerSpawns"] = f"{len(led)}"
    return out


if __name__ == "__main__":
    run_main(build)
