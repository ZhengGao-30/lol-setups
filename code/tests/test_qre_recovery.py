"""Synthetic checks for lolsetup.qre: (1) the solver returns a fixed point; (2) NPL recovers team parameters
(lam, nu, kappa) from choices simulated at the true QRE of each contest."""
import numpy as np

from lolsetup import qre


def random_contest(rng, oc):
    n_i, n_j = rng.choice([4, 5], p=[0.15, 0.85]), rng.choice([4, 5], p=[0.15, 0.85])
    obj = rng.uniform(0.02, 0.12)
    v = qre.Values(obj_self=obj, obj_opp=-obj * rng.uniform(0.8, 1.2), fight_win=rng.uniform(0.02, 0.08),
                   fight_lose=-rng.uniform(0.02, 0.08), front_win=rng.uniform(0.01, 0.05),
                   front_lose=-rng.uniform(0.01, 0.05))
    return qre.build_game(n_i, n_j, v, rng.normal(0, 0.1), oc)[0]


def main(n_contests=400, n_teams=4, seed=0):
    rng = np.random.default_rng(seed)
    oc = qre.Outcome(gamma_n=2.0, gamma_g=3.0, delta_tau=0.3, gamma_front=1.0)
    true = np.array([[40.0, 1.0, 0.5], [80.0, 1.0, 1.0], [40.0, 1.6, 0.5], [120.0, 0.8, 1.5]])[:n_teams]
    games, choices = [], []
    for _ in range(n_contests):
        g = random_contest(rng, oc)
        ti, tj = rng.choice(n_teams, 2, replace=False)
        si, sj, it = qre.solve(g, tuple(true[ti]), tuple(true[tj]))
        Oi, Ri, taui, Oj, Rj, tauj = g[:6]
        # fixed-point check: each strategy is the logit response to the other
        ri = np.exp(qre.utilities(Oi, Ri, sj, taui, *true[ti]))
        assert np.abs(ri / ri.sum() - si).max() < 1e-6, "solver did not reach the fixed point"
        games.append((g, (ti, tj)))
        choices.append((rng.choice(len(si), p=si), rng.choice(len(sj), p=sj)))
    est, _, hist = qre.npl(games, choices, None, n_teams)
    print("NPL iterations:", len(hist), "final nll:", round(hist[-1][0], 2))
    print("team | lam true est | nu true est | kappa true est")
    for t in range(n_teams):
        print(t, "|", true[t, 0], round(est[t, 0], 1), "|", true[t, 1], round(est[t, 1], 2), "|", true[t, 2], round(est[t, 2], 2))
    return true, est


if __name__ == "__main__":
    main()
