"""Fig. 4: the outcome-only win surface W and the objective values it implies. (a) calibration by decile
(pooled specification), (b) log-loss by phase for the three W specifications, (c) objective values at W = 0.5
(win probability added over 180 s vs total effect on the result)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

ORDER = ["HORDE", "RIFTHERALD", "DRAGON_1", "DRAGON_2", "DRAGON_3", "DRAGON_4", "ATAKHAN", "BARON_NASHOR", "ELDER",
         "FIGHT", "TOWER"]
LAB = {"HORDE": "grubs", "RIFTHERALD": "herald", "DRAGON_1": "dragon 1", "DRAGON_2": "dragon 2",
       "DRAGON_3": "dragon 3 (soul point)", "DRAGON_4": "dragon 4 (soul)", "ATAKHAN": "Atakhan",
       "BARON_NASHOR": "baron", "ELDER": "elder", "FIGHT": "fight (+1 kill)", "TOWER": "tower"}


def build(N):
    cal = csv("results/winsurface/pooled/calibration_deciles.csv")
    ph = {k: csv(f"results/winsurface/{k}/fit_by_phase.csv") for k in ("pooled", "r2509", "all")}
    ev = csv("results/values/event_values.csv")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 2.35), gridspec_kw=dict(width_ratios=[0.9, 0.9, 1.6], wspace=0.55))
    a = axs[0]
    a.plot([0, 1], [0, 1], color=C["grey"], lw=0.6, ls=":")
    a.plot(cal.pred, cal.obs, "o-", ms=3, color=C["blue"])
    a.set_xlabel("predicted $W$")
    a.set_ylabel("observed win rate")
    a.set_title("(a) calibration || out of fold, deciles")
    b = axs[1]
    x = np.arange(3)
    for i, (k, lab) in enumerate((("pooled", "pooled (main)"), ("r2509", "same rules"), ("all", "all games"))):
        b.bar(x + (i - 1) * 0.27, ph[k].logloss, 0.27, label=lab)
    b.set_xticks(x, ["1–13", "14–24", "25+"])
    b.set_xlabel("game minute")
    b.set_ylabel("log-loss")
    b.legend(fontsize=6.3, loc="upper right")
    b.set_title("(b) log-loss || by game phase")
    c = axs[2]
    col = "value_at_W=0.5 (points)"
    v = ev.groupby(["kind", "type"])[col].first().unstack(0).reindex(ORDER)
    y = np.arange(len(v))
    c.barh(y + 0.2, v["wpa"], 0.4, color=C["blue"], label="WPA over 180 s (main)")
    c.barh(y - 0.2, v["result"], 0.4, color=C["orange"], label="total effect on result")
    c.set_yticks(y, [LAB[t] for t in v.index])
    c.invert_yaxis()
    c.set_xlabel("value at $W=0.5$ (win-prob. points)")
    c.legend(fontsize=6.3, loc="lower right")
    c.set_title("(c) event values || win probability added, at W = 0.5")
    dev = (cal.pred - cal.obs).abs().max()
    N["wCalMaxDevPP"] = f(dev, 1, pct=True)
    for k in ("pooled", "r2509", "all"):
        for i, nm in enumerate(("Early", "Mid", "Late")):
            N[f"wLogloss{nm}{k.capitalize()}"] = f"{ph[k].logloss.iloc[i]:.3f}"
    for t in ORDER:
        key = t.replace("_", "").replace("1", "One").replace("2", "Two").replace("3", "Three").replace("4", "Four")
        N[f"val{key}Wpa"] = f(v.loc[t, "wpa"], 1)
        N[f"val{key}Res"] = f(v.loc[t, "result"], 1)
    return save(fig, "fig04_values")


if __name__ == "__main__":
    run_main(build)
