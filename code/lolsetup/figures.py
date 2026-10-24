"""Figures from the result tables (each function skips when its inputs are missing).

    python -m lolsetup.figures      -> results/figures/*.png (+ .pdf)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io

R = io.RESULTS
OUT = R / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
C = {"det": "#1b6ca8", "interp": "#d98324", "hold": "#b0b0b0", "pin": "#3a8a4a", "T1": "#c0392b", "other": "#7f8c8d"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def fig_tracker():
    f = R / "tracker" / "eval_pooled.parquet"
    if not f.exists():
        return
    d = pd.read_parquet(f)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    xs = np.logspace(1.5, 4.1, 200)
    for st in ("det", "interp", "hold"):
        e = d.loc[d["state"] == st, "err"].dropna().to_numpy()
        if len(e) > 20:
            ax[0].plot(xs, [(e <= x).mean() for x in xs], color=C[st], label=f"{st} (n={len(e)})")
    ax[0].plot(xs, [(d["err"].dropna() <= x).mean() for x in xs], "k", lw=1.5, label="all")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("position error (map units)")
    ax[0].set_ylabel("cumulative share")
    ax[0].legend(frameon=False)
    ax[0].axvline(600, color="k", ls=":", lw=0.7)
    cr = pd.read_csv(R / "tracker" / "summary_by_crowding.csv")
    ax[1].bar(cr["crowd_bin"].astype(str), cr["lt600"], color="#1b6ca8")
    ax[1].set_xlabel("other champions within 2000 units")
    ax[1].set_ylabel("share of positions with error < 600")
    pg = pd.read_csv(R / "tracker" / "summary_per_game.csv")
    ax[2].hist(pg["lt600"], bins=20, color="#1b6ca8")
    ax[2].set_xlabel("per-game share with error < 600")
    ax[2].set_ylabel("games")
    fig.suptitle("Tracker validation against Riot V5 positions (odd minutes)", y=1.02)
    save(fig, "fig02_tracker_validation")


def fig_baseline():
    f = R / "baseline" / "tagnet_k_selection.csv"
    if not f.exists():
        return
    ks = pd.read_csv(f)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for v, c in (("noctx", "#1b6ca8"), ("ctx", "#c0392b")):
        s = ks[ks["variant"] == v]
        ax[0].plot(s["K"], s["silhouette"], color=c, label=v)
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("silhouette (TAGNet-style K-Means)")
    ax[0].legend(frameon=False)
    ps = pd.read_csv(R / "baseline" / "tagnet_per_seed.csv")
    ax[1].boxplot([ps[ps["variant"] == v]["pairwise_ari_mean"].dropna() for v in ("noctx", "ctx")], tick_labels=["no context", "with context"])
    ax[1].set_ylabel("mean cross-seed ARI")
    for i, v in enumerate(("noctx", "ctx")):
        x = np.load(R / "baseline" / f"tagnet_{v}_bootstrap_ari.npy")
        ax[2].hist(x, bins=25, alpha=0.6, color=("#1b6ca8", "#c0392b")[i], label=v)
    ax[2].set_xlabel("series-bootstrap ARI of K-Means (K=6)")
    ax[2].legend(frameon=False)
    fig.suptitle("TAGNet baseline on the V layer: stability", y=1.02)
    save(fig, "fig_baseline_tagnet_stability")


def fig_fights():
    f = R / "fights" / "s7_binned.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s["p_victim_larger"], 0.4, color="#1b6ca8", label="observed")
    ax.bar(x + 0.2, s["square_law"], 0.4, color="#bbbbbb", label="square law")
    ax.set_xticks(x)
    ax.set_xticklabels(s["ratio"])
    ax.set_ylabel("P(next death on the larger side)")
    ax.legend(frameon=False)
    save(fig, "fig10_fight_s7")


def fig_hasse(tag="_v"):
    f = R / f"hasse{tag}" / "selection.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(s["K"], s["test_loglik_per_setup"], "o-", color="#1b6ca8")
    ax[0].set_ylabel("held-out log-lik per setup")
    ax[1].plot(s["K"], s["bic"], "o-", color="#d98324", label="BIC")
    ax[1].plot(s["K"], s["icl"], "s-", color="#c0392b", label="ICL")
    ax[1].legend(frameon=False)
    ax[1].set_ylabel("criterion (lower is better)")
    for a in ax[:2]:
        a.set_xlabel("K")
    bp = R / f"hasse{tag}" / "best_patterns.json"
    if bp.exists():
        pat = json.loads(bp.read_text())["patterns"]
        ax[2].bar(range(len(pat)), [p["theta"] for p in pat], color="#1b6ca8")
        ax[2].set_xlabel("pattern")
        ax[2].set_ylabel("rigidity theta (Hasse steps^-1)")
    save(fig, "fig04_hasse_selection")


def fig_qre():
    f = R / "robustness" / "qre_bootstrap_ci.csv"
    if f.exists():
        d = pd.read_csv(f)
        s = d[d["quantity"].str.startswith("shap_")]
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        lab = {"shap_L": "Lanchester", "shap_C": "curvature", "shap_F": "second front", "shap_B": "behavioural"}
        y = np.arange(len(s))
        ax.errorbar(s["point"], y, xerr=[s["point"] - s["lo"], s["hi"] - s["point"]], fmt="o", color="#1b6ca8")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([lab.get(q, q) for q in s["quantity"]])
        ax.set_xlabel("Shapley share of the trailing-leading commitment gap")
        save(fig, "fig08_qre_shapley")
    f = R / "robustness" / "qre_team_parameter_ci.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        for a, p in zip(ax, ("lam", "kappa")):
            s = d[(d["param"] == p) & (d["team"] != "other")].sort_values("point")
            y = np.arange(len(s))
            a.errorbar(s["point"], y, xerr=[(s["point"] - s["lo"]).clip(lower=0), (s["hi"] - s["point"]).clip(lower=0)], fmt="o",
                       color="#1b6ca8", ecolor="#9bb7d4")
            for yi, t in zip(y, s["team"]):
                if t == "T1":
                    a.plot(s.loc[s["team"] == "T1", "point"], yi, "o", color=C["T1"], ms=8)
            a.set_yticks(y)
            a.set_yticklabels(s["team"], fontsize=7)
            a.set_xlabel(p)
        save(fig, "fig06b_team_parameters")
    f = R / "robustness" / "qre_variants.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(d))
        for i, k in enumerate("LCFB"):
            ax.bar(x + (i - 1.5) * 0.18, d[f"shap_{k}"], 0.18, label=k)
        ax.set_xticks(x)
        ax.set_xticklabels(d["variant"], rotation=45, ha="right")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(frameon=False, ncol=4)
        ax.set_ylabel("Shapley value")
        save(fig, "fig09_qre_variants")


def fig_context():
    f = R / "context" / "ame_v.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    feats = list(d["feature"].unique())
    pats = sorted(d["pattern"].unique())
    fig, axes = plt.subplots(1, len(feats), figsize=(2.0 * len(feats), 3.4), sharey=True)
    for ax, ft in zip(np.atleast_1d(axes), feats):
        s = d[d["feature"] == ft].sort_values("pattern")
        ax.errorbar(s["ame_per_sd"], s["pattern"], xerr=[s["ame_per_sd"] - s["lo"], s["hi"] - s["ame_per_sd"]], fmt="o", color="#1b6ca8")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(ft, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("pattern")
    fig.suptitle("Context effects on pattern occupancy (AME per SD, series bootstrap 95% CI)", y=1.02)
    save(fig, "fig07_context_forest")


def fig_fingerprints():
    f = R / "fingerprints" / "usage.csv"
    if not f.exists():
        return
    u = pd.read_csv(f, index_col=0)
    occ = [c for c in u.columns if c.startswith("occ_")]
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(occ), 0.28 * len(u) + 1))
    im = ax.imshow(u[occ].to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(occ)))
    ax.set_xticklabels([c.replace("occ_pattern", "P") for c in occ])
    ax.set_yticks(range(len(u)))
    ax.set_yticklabels(u.index, fontsize=7)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "T1":
            lab.set_color(C["T1"])
            lab.set_fontweight("bold")
    fig.colorbar(im, ax=ax, label="occupancy share")
    save(fig, "fig06_fingerprints")


def fig_t1():
    f = R / "t1" / "A3_faker_vs_other_mids.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        y = np.arange(len(d))
        ax.errorbar(d["diff"], y, xerr=[d["diff"] - d["lo"], d["hi"] - d["diff"]], fmt="o", color=C["T1"])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(d["measure"])
        ax.set_xlabel("Faker minus other starting mid laners (raw units)")
        save(fig, "figA3_faker_positioning")


def main():
    for fn in (fig_tracker, fig_baseline, fig_fights, fig_hasse, fig_qre, fig_context, fig_fingerprints, fig_t1):
        try:
            fn()
        except Exception as e:  # a missing/partial table must not stop the other figures
            print("skipped", fn.__name__, repr(e))


if __name__ == "__main__":
    main()
