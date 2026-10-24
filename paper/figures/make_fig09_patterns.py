"""Fig. 9 (and appendix variants): the HASSE pattern dictionary. Each row is a pattern: its centre path on
Pi_5 over the 13 snapshots (s-120..s+60), coloured by the number of blocks, with rigidity theta, inertia iota and
the mixture weight in each objective class."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

CLASSES = ["early river", "Atakhan", "baron/elder"]


def draw(j, name, N=None, tag=""):
    P = j["patterns"]
    K = len(P)
    T = len(P[0]["centre_path"])
    offs = np.arange(T) * 15 - 120
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_W, 0.55 * K + 0.8), gridspec_kw=dict(width_ratios=[4.2, 1], wspace=0.05))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gather", ["#2F5D96", "#7FA1CE", "#C9D7EC", "#EFE5CB", "#F9F5EA"], N=5)
    for i, p in enumerate(P):
        for t, part in enumerate(p["centre_path"]):
            part = part.replace(" ", "")
            nb = len(part.split("|"))
            a.add_patch(plt.Rectangle((t, K - 1 - i), 1, 1, color=cmap(nb - 1), alpha=0.9, lw=0.4, ec="white"))
            a.text(t + 0.5, K - 1 - i + 0.5, "\n".join(part.split("|")), ha="center", va="center", fontsize=5.6,
                   linespacing=0.95, color="#1d1d1d" if nb > 1 else "white")
        a.text(-0.2, K - 1 - i + 0.5, f"P{p['pattern']}  $\\theta$={p['theta']:.2f} $\\iota$={p['iota']:.2f}",
               ha="right", va="center", fontsize=6.3)
    a.set_xlim(0, T)
    a.set_ylim(0, K)
    a.set_xticks(np.arange(T) + 0.5, [str(o) if o % 60 == 0 else "" for o in offs], fontsize=6.5)
    a.set_yticks([])
    a.axvline(8, color="k", lw=0.8)
    a.axvline(9, color="k", lw=0.8)
    a.set_xlabel("s from spawn (centre path $\\mu_{c,t}$; boxed column = spawn)")
    for sp in ("left", "bottom"):
        a.spines[sp].set_visible(False)
    W = np.array([p["weight_by_class"] for p in P])
    y = np.arange(K)[::-1]
    for c in range(3):
        b.barh(y + 0.5 + (c - 1) * 0.26, W[:, c], 0.26, color=[C["sky"], C["orange"], C["red"]][c], label=CLASSES[c])
    b.set_ylim(0, K)
    b.set_xlim(0, max(0.6, 2.0 * W.max()))
    b.set_yticks([])
    b.set_xlabel("weight $w_{class}$")
    b.legend(fontsize=6.0, loc="upper right", handlelength=0.9, ncol=1, borderaxespad=0.1)
    if N is not None:
        N[f"xi{tag}"] = f"{j['xi']:.3f}"
        N[f"thetaMin{tag}"] = f"{min(p['theta'] for p in P):.2f}"
        N[f"thetaMax{tag}"] = f"{max(p['theta'] for p in P):.2f}"
        N[f"iotaMin{tag}"] = f"{min(p['iota'] for p in P):.2f}"
        N[f"iotaMax{tag}"] = f"{max(p['iota'] for p in P):.2f}"
    return save(fig, name)


def build(N):
    out = [draw(js("results/hasse_v/patterns_K6.json"), "fig09_patterns", N, "Six")]
    for K in (5, 8):
        try:
            out.append(draw(js(f"results/hasse_v/patterns_K{K}.json"), f"figA_patterns_K{K}", N, {5: "Five", 8: "Eight"}[K]))
        except Missing:
            pass
    j = js("results/hasse_v/patterns_K6.json")
    for p in j["patterns"]:
        i = p["pattern"]
        nm = ["Zero", "One", "Two", "Three", "Four", "Five"][i]
        N[f"pat{nm}Theta"] = f"{p['theta']:.2f}"
        N[f"pat{nm}Iota"] = f"{p['iota']:.2f}"
        for c, cn in enumerate(("Early", "Atakhan", "Late")):
            N[f"pat{nm}W{cn}"] = f"{p['weight_by_class'][c]:.2f}"
        N[f"pat{nm}AtSpawn"] = p["centre_path"][8].replace(" ", "")
    return out


if __name__ == "__main__":
    run_main(build)
