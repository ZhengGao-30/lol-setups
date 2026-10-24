"""Fig. 2: measurement validation. (a) broadcast-clock alignment, (b) tracker error ECDF by position source,
(c) error by local crowding, (d) formation agreement with Riot V5 by alive count."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403


def build(N):
    games = csv("results/clock/games.csv")
    seg = csv("results/clock/segments.csv")
    ev = pd.read_parquet(src("results/tracker/eval_pooled.parquet"))
    crowd = csv("results/tracker/summary_by_crowding.csv")
    form = csv("results/tracker/summary_formation.csv")
    pooled = csv("results/tracker/summary_pooled.csv").iloc[0]
    st = csv("results/tracker/summary_by_state.csv").set_index("state")
    pg = csv("results/tracker/summary_per_game.csv")

    fig, axs = plt.subplots(2, 2, figsize=(TEXT_W, 3.5), gridspec_kw=dict(wspace=0.32, hspace=0.78))
    axs = axs.ravel()
    a = axs[0]
    a.hist(games.end_diff_s.clip(-20, 2), bins=np.arange(-20, 2.5, 0.5), color=C["blue"])
    a.set_xlabel("last live clock $-$ game length (s)")
    a.set_ylabel("games")
    a.set_title("(a) clock alignment || last live reading vs official game end")

    b = axs[1]
    lab = {"det": "detected", "interp": "interpolated", "hold": "held"}
    for s_, col in zip(("det", "interp", "hold"), (C["blue"], C["orange"], C["red"])):
        e = np.sort(ev.loc[ev.state == s_, "err"].dropna().to_numpy())
        b.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, label=f"{lab[s_]} ({100 * st.loc[s_, 'share']:.0f}%)")
    e = np.sort(ev.err.dropna().to_numpy())
    b.plot(e, np.arange(1, len(e) + 1) / len(e), color="k", lw=0.8, ls="--", label="pooled")
    b.set_xscale("log")
    b.axvline(600, color=C["grey"], lw=0.6, ls=":")
    b.set_xlabel("position error (map units)")
    b.set_ylabel("ECDF")
    b.legend(fontsize=6.4, loc="upper left")
    b.set_title("(b) tracker error || against Riot positions, by source")

    c = axs[2]
    c.bar(crowd.crowd_bin.astype(str), crowd.lt600, color=C["sky"])
    c.set_xlabel("other champions within 2000")
    c.set_ylabel("share error $<$ 600")
    c.set_ylim(0, 1)
    c.set_title("(c) crowding || errors grow with nearby champions")

    d = axs[3]
    fm = form[form.n_alive.astype(str).isin(["2", "3", "4", "5"])].drop_duplicates("n_alive")
    x = np.arange(len(fm))
    d.bar(x - 0.18, fm.exact, 0.36, color=C["blue"], label="exact match")
    d.bar(x + 0.18, fm.rand, 0.36, color=C["orange"], label="Rand index")
    d.set_xticks(x, fm.n_alive.astype(str))
    d.set_xlabel("alive players")
    d.set_ylim(0, 1)
    d.set_ylabel("agreement with V5")
    d.legend(fontsize=6.4, loc="upper right", ncol=2)
    d.set_ylim(0, 1.15)
    d.set_title("(d) formations || tracked vs Riot partitions")

    N["clockGames"] = f"{len(games)}"
    N["clockReplaySeg"] = f"{int((~seg.live).sum())}"
    N["clockLiveMedPct"] = f(games.frac_live.median(), 1, pct=True)
    N["clockMisreadMedPct"] = f(games.misread_rate.median(), 2, pct=True)
    N["clockEndDiffMed"] = f(games.end_diff_s.median(), 1)
    N["trkN"] = f"{int(pooled.n):,}"
    N["trkMed"] = f"{pooled.q50:,.0f}"
    N["trkPninety"] = f"{pooled.q90:,.0f}"
    N["trkPninetyfive"] = f"{pooled.q95:,.0f}"
    N["trkLtThree"] = f"{pooled.lt300:.3f}"
    N["trkLtSix"] = f"{pooled.lt600:.3f}"
    N["trkLtTen"] = f"{pooled.lt1000:.3f}"
    N["trkDetShare"] = f(st.loc["det", "share"], 0, pct=True)
    N["trkDetMed"] = f"{st.loc['det', 'q50']:,.0f}"
    N["trkDetLtSix"] = f"{st.loc['det', 'lt600']:.3f}"
    N["trkDetPninetyfive"] = f"{st.loc['det', 'q95']:,.0f}"
    N["trkInterpShare"] = f(st.loc["interp", "share"], 0, pct=True)
    N["trkInterpMed"] = f"{st.loc['interp', 'q50']:,.0f}"
    N["trkInterpLtSix"] = f"{st.loc['interp', 'lt600']:.3f}"
    N["trkHoldMed"] = f"{st.loc['hold', 'q50']:,.0f}"
    cz = crowd.set_index(crowd.crowd_bin.astype(str))
    N["trkCrowdZeroLtSix"] = f"{cz.loc['0', 'lt600']:.3f}"
    N["trkCrowdFourLtSix"] = f"{cz.loc['4+', 'lt600']:.3f}"
    allrow = form[form.n_alive.astype(str) == "all"].iloc[0]
    N["formExact"] = f"{allrow.exact:.3f}"
    N["formRand"] = f"{allrow.rand:.3f}"
    five = form[form.n_alive.astype(str) == "5"].iloc[0]
    N["formExactFive"] = f"{five.exact:.3f}"
    N["trkGameMedMin"] = f"{pg.q50.min():,.0f}"
    N["trkGameMedMax"] = f"{pg.q50.max():,.0f}"
    N["trkGamesBelowSix"] = f"{int((pg.lt600 < 0.6).sum())}"
    return save(fig, "fig02_measurement")


if __name__ == "__main__":
    run_main(build)
