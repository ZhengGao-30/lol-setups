"""Diagnostics of the QRE data and likelihood surface (why does the fit collapse to lam -> 0?)."""
import numpy as np
import pandas as pd

from . import io, qre, run_qre


def main():
    P = run_qre.prepare("wpa")
    info, games, choices, spec, st_b = P["info"], P["games"], P["choices"], P["spec"], P["st_b"]
    print("contests", len(games))
    print(info[[c for c in info.columns if c.startswith("v_")] + ["W_blue", "minute", "gd", "lng"]].describe().round(4).to_string())
    ks = []
    for s, (a, b) in zip(spec, choices):
        ks.append((qre.actions(s["n_b"], s["kb"])[a], qre.actions(s["n_r"], s["kr"])[b]))
    kb = np.array([x[0] for x in ks]); kr = np.array([x[1] for x in ks])
    print("blue k distribution", np.bincount(kb[:, 0], minlength=6), "red", np.bincount(kr[:, 0], minlength=6))
    print("blue m", np.bincount(kb[:, 1], minlength=6), "tau share", kb[:, 2].mean(), kr[:, 2].mean())
    print("n_b", np.bincount(info["n_b"], minlength=6), "n_r", np.bincount(info["n_r"], minlength=6))
    print("by objective: commit share (k>=3) blue", pd.Series(kb[:, 0] >= 3).groupby(info["objective"].to_numpy()).mean().round(3).to_dict())
    print("by state: commit share", pd.Series(kb[:, 0] >= 3).groupby(st_b).mean().round(3).to_dict())
    # pseudo-likelihood vs lam with uniform opponent, global parameters only (nu = 1, kappa = 0)
    strat = [(np.full(g[0].shape[0], 1 / g[0].shape[0]), np.full(g[3].shape[0], 1 / g[3].shape[0])) for g, _ in games]
    data = qre._pseudo_data(games, strat, choices)
    a, b, tau, mask, y, team = data
    for nu in (1.0, 3.0):
        for lam in (0.0, 1.0, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0):
            for kap in (0.0, 1.0):
                u = np.where(mask, lam * (nu * a + b) - kap * tau, -np.inf)
                lse = np.logaddexp.reduce(u, axis=1)
                ll = (u[np.arange(len(y)), y] - lse).sum()
                print(f"nu {nu} lam {lam:7.1f} kappa {kap}: pseudo-ll {ll:.1f}")
    # does the payoff of the chosen action exceed the mean payoff?
    pay = (a + b)[np.arange(len(y)), y] - np.array([(a + b)[i, mask[i]].mean() for i in range(len(y))])
    print("mean (payoff of chosen - mean payoff over own actions), objective+rest:", pay.mean())
    pay_o = a[np.arange(len(y)), y] - np.array([a[i, mask[i]].mean() for i in range(len(y))])
    print("objective part only:", pay_o.mean(), " tau part (chosen tau - mean tau):",
          (tau[np.arange(len(y)), y] - np.array([tau[i, mask[i]].mean() for i in range(len(y))])).mean())
    best = np.array([np.argmax(np.where(mask[i], a[i] + b[i], -np.inf)) for i in range(len(y))])
    kbest = np.array([qre.actions(5)[x][0] if True else 0 for x in best])
    print("payoff-maximising action's k distribution (uniform opp, n=5 indices only approx):", np.bincount(kbest, minlength=6))


if __name__ == "__main__":
    main()
