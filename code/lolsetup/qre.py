"""LB-QRE: objective contests as a two-team logit quantal response equilibrium (explanation layer).

Actions for a team with n alive players: a = (k, m, tau) with k players at the pit, m on the second front,
k + m <= n, and tau = 1 if the pit group was in position before the spawn (only when k > 0); 36 actions for n = 5.

Outcome model (every component is estimated outside the choice model; nothing here sees setup choices):
  contest   both teams commit (k_i, k_j > 0): team i wins the fight with
                p_i = Lambda(gamma_n ln(k_i / k_j) + gamma_g ln(g_i / g_j) + delta_tau (tau_i - tau_j))
            (Lanchester -> Tullock; analysis_fights) and the fight winner takes the objective;
            only i commits: i takes the objective; nobody commits: no change within the window.
  front     m_i vs m_j: i wins the second front with q_i = m_i^g2 / (m_i^g2 + m_j^g2) (no event when both are 0).
  payoff    U_i(a_i, a_j) = sum over outcomes P(outcome) * [W_i(x + Delta_outcome) - W_i(x)], in win-probability
            points, with the state values v (see Values) read off the outcome-only win surface W (winsurface.py).
Choice: sigma_i(a) proportional to exp(lam_i * [nu_i * U_obj_i(a, sigma_j) + U_rest_i(a, sigma_j)] - kappa_i * tau(a)):
  lam = decision precision (lam -> inf nests Nash), nu = valuation multiplier on the objective (nu = 1:
  pure win-probability maximisation, H_winmax), kappa = cost of early positioning in utils.
With nu = 1 and own-action costs only the game is zero-sum up to own-action terms, so the logit QRE is the unique
saddle point of x'Uy + H(x)/lam_i - H(y)/lam_j; it is computed by damped fixed-point iteration.
Estimation: nested pseudo-likelihood (Aguirregabiria & Mira 2007): with the opponents' strategies fixed at the
current estimate the likelihood is a conditional logit in theta; theta and the strategies are updated in turn.
"""
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import expit, log_softmax, softmax


def actions(n=5, kmax=None):
    """List of (k, m, tau) for n alive players; k <= kmax (feasibility: players that can reach the pit in time)."""
    out = []
    for k in range(n + 1 if kmax is None else min(n, kmax) + 1):
        for m in range(n - k + 1):
            out.append((k, m, 0))
            if k > 0:
                out.append((k, m, 1))
    return np.array(out)


@dataclass
class Values:
    """State values for team i in win-probability points (all relative to the current state x):
    obj_self / obj_opp: objective taken by i / by j; fight_win / fight_lose: i wins / loses the fight at the pit;
    front_win / front_lose: i wins / loses the second front."""
    obj_self: float
    obj_opp: float
    fight_win: float
    fight_lose: float
    front_win: float
    front_lose: float

    def flip(self):
        """The same state from the opponent's side (antisymmetric W: every value changes sign and swaps)."""
        return Values(-self.obj_opp, -self.obj_self, -self.fight_lose, -self.fight_win, -self.front_lose, -self.front_win)


@dataclass
class Outcome:
    gamma_n: float
    gamma_g: float
    delta_tau: float
    gamma_front: float


def payoff_parts(A_i, A_j, v, ln_gold_ratio, oc):
    """Payoff matrices (objective part, rest) for team i: rows = i's actions, columns = j's actions."""
    ki, mi, ti = (A_i[:, c][:, None].astype(float) for c in range(3))
    kj, mj, tj = (A_j[:, c][None, :].astype(float) for c in range(3))
    both = (ki > 0) & (kj > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lin = oc.gamma_n * np.log(np.where(both, ki / np.where(kj > 0, kj, 1), 1.0)) + oc.gamma_g * ln_gold_ratio \
            + oc.delta_tau * (ti - tj)
    p = np.where(both, expit(lin), np.where(ki > 0, 1.0, 0.0))  # P(i wins the pit), i alone wins it
    p_opp = np.where(both, 1 - p, np.where(kj > 0, 1.0, 0.0))
    U_obj = p * v.obj_self + p_opp * v.obj_opp
    U_fight = np.where(both, p * v.fight_win + (1 - p) * v.fight_lose, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where((mi + mj) > 0, mi ** oc.gamma_front / (mi ** oc.gamma_front + mj ** oc.gamma_front), np.nan)
    U_front = np.where((mi + mj) > 0, q * v.front_win + (1 - q) * v.front_lose, 0.0)
    return U_obj, U_fight + U_front


def _pen2(pen):
    pen = np.asarray(pen, float)
    return pen[:, None] if pen.ndim == 1 else pen


def utilities(U_obj, U_rest, sigma_opp, pen, lam, nu, *cost):
    """Choice utilities of all own actions against the opponent's mixed strategy. pen: (A,) tau, or (A, m) with
    columns [tau, travel cost, ...]; cost = the matching coefficients (kappa, psi, ...)."""
    pen = _pen2(pen)
    return lam * (nu * U_obj @ sigma_opp + U_rest @ sigma_opp) - pen @ np.array(cost[:pen.shape[1]], float)


def solve(game, th_i, th_j, tol=1e-10, max_iter=5000, damp=0.5):
    """Logit QRE of one state. game = (U_obj_i, U_rest_i, tau_i, U_obj_j, U_rest_j, tau_j); th = (lam, nu, kappa)."""
    Oi, Ri, ti, Oj, Rj, tj = game
    li = np.zeros(Oi.shape[0]) - np.log(Oi.shape[0])
    lj = np.zeros(Oj.shape[0]) - np.log(Oj.shape[0])
    for it in range(max_iter):
        ni = log_softmax(utilities(Oi, Ri, np.exp(lj), ti, *th_i))
        nj = log_softmax(utilities(Oj, Rj, np.exp(li), tj, *th_j))
        ni = np.logaddexp(np.log(damp) + ni, np.log(1 - damp) + li)
        nj = np.logaddexp(np.log(damp) + nj, np.log(1 - damp) + lj)
        err = max(np.abs(np.exp(ni) - np.exp(li)).max(), np.abs(np.exp(nj) - np.exp(lj)).max())
        li, lj = ni - np.logaddexp.reduce(ni), nj - np.logaddexp.reduce(nj)
        if err < tol:
            break
    return np.exp(li), np.exp(lj), it


def build_game(n_i, n_j, v_i, ln_gold_ratio, oc, kmax_i=None, kmax_j=None, travel_i=None, travel_j=None):
    """All matrices of one contest from team i's values (team j's are the antisymmetric flip). travel_x[k] = travel
    time (minutes) of the k players of team x that are nearest to the pit at the spawn. Penalty columns of an action:
    [tau, travel(k), k, m] with coefficients (kappa, psi, omega_k, omega_m): early positioning, travel time, and a
    linear opportunity cost of every player committed to the pit / to the second front."""
    A_i, A_j = actions(n_i, kmax_i), actions(n_j, kmax_j)
    Oi, Ri = payoff_parts(A_i, A_j, v_i, ln_gold_ratio, oc)
    Oj, Rj = payoff_parts(A_j, A_i, v_i.flip(), -ln_gold_ratio, oc)

    def pen(A, travel):
        tau = A[:, 2].astype(float)
        return tau if travel is None else np.column_stack([tau, np.asarray(travel, float)[A[:, 0]], A[:, 0], A[:, 1]])
    return (Oi, Ri, pen(A_i, travel_i), Oj, Rj, pen(A_j, travel_j)), A_i, A_j


def _pseudo_data(games, strat, choices):
    """Stack both sides of every contest against the opponent's current strategy: expected objective and
    other payoffs per own action (padded to the largest action set), penalty columns, chosen action and team index."""
    rows = []
    for (g, (ti, tj)), (si, sj), (ai, aj) in zip(games, strat, choices):
        Oi, Ri, peni, Oj, Rj, penj = g
        rows.append((Oi @ sj, Ri @ sj, _pen2(peni), ai, ti))
        rows.append((Oj @ si, Rj @ si, _pen2(penj), aj, tj))
    n, A, m = len(rows), max(len(r[0]) for r in rows), rows[0][2].shape[1]
    a, b, pen, mask = np.zeros((n, A)), np.zeros((n, A)), np.zeros((n, A, m)), np.zeros((n, A), bool)
    for i, (x, y_, t, _, _) in enumerate(rows):
        a[i, :len(x)], b[i, :len(x)], pen[i, :len(x)], mask[i, :len(x)] = x, y_, t, True
    return a, b, pen, mask, np.array([r[3] for r in rows]), np.array([r[4] for r in rows])


SHRINK = np.array([1.0, 0.5, 1.0, 1.0, 0.5, 0.5])  # prior sd of team deviations in (log lam, log nu, kappa, psi, omega_k, omega_m)
GLOBAL_LOGNU_SD = 2.0  # weak prior on the global log nu: keeps the lam-nu ridge finite when the non-objective
# payoffs carry no information (lam -> 0, nu -> inf with lam * nu fixed); report lam_obj = lam * nu, lam_rest = lam


def _team_params(flat, n_teams):
    """flat = [global (P), deviations (n_teams x P)] with P = 2 + number of penalty columns -> per-team array."""
    P = len(flat) // (n_teams + 1)
    return flat[:P][None, :] + flat[P:].reshape(n_teams, P)


def _probs(t, data):
    a, b, pen, mask, y, team = data
    lam, nu = np.exp(t[team, 0])[:, None], np.exp(t[team, 1])[:, None]
    s = nu * a + b
    u = np.where(mask, lam * s - np.einsum("nam,nm->na", pen, t[team, 2:]), -np.inf)
    lse = np.logaddexp.reduce(u, axis=1)
    return u, lse, np.exp(u - lse[:, None]), lam, nu, s


def _nll_grad(flat, data, n_teams):
    """Penalised pseudo-likelihood (conditional logit with the opponents' strategies fixed; Gaussian shrinkage of
    team deviations towards the global parameters) and its analytic gradient."""
    a, b, pen, mask, y, team = data
    P = len(flat) // (n_teams + 1)
    t = _team_params(flat, n_teams)
    u, lse, p, lam, nu, s = _probs(t, data)
    i = np.arange(len(y))
    dev = flat[P:].reshape(n_teams, P)
    shrink = SHRINK[:P]
    nll = -(u[i, y] - lse).sum() + 0.5 * ((dev / shrink) ** 2).sum() + 0.5 * (flat[1] / GLOBAL_LOGNU_SD) ** 2
    g = np.zeros((n_teams, P))
    np.add.at(g[:, 0], team, -(lam[:, 0] * s[i, y] - (p * lam * s).sum(1)))
    np.add.at(g[:, 1], team, -(lam[:, 0] * nu[:, 0] * a[i, y] - (p * lam * nu * a).sum(1)))
    for j in range(P - 2):
        np.add.at(g[:, 2 + j], team, -(-pen[i, y, j] + (p * pen[:, :, j]).sum(1)))
    g_glob = g.sum(0)
    g_glob[1] += flat[1] / GLOBAL_LOGNU_SD ** 2
    return nll, np.r_[g_glob, (g + dev / shrink ** 2).ravel()]


def npl(games, choices, team_idx, n_teams, n_iter=50, theta0=None, tol=1e-5, fix_nu=False):
    """Nested pseudo-likelihood with partial pooling. games: list of (game, (team_i, team_j)); choices: list of
    (a_i index, a_j index). Team parameters (log lam, log nu, kappa[, psi]) = global + deviation, deviations shrunk with
    prior sd SHRINK. Each step maximises the penalised pseudo-likelihood with the opponents' strategies fixed, then
    sets every strategy to the logit response at the new parameters (the NPL operator Psi(theta, P)).
    Returns (theta (n_teams, P) as (lam, nu, kappa[, psi]), strategies, history [(objective, max parameter change)])."""
    P = 2 + _pen2(games[0][0][2]).shape[1]
    flat = np.r_[np.log(5.0), np.zeros(P - 1), np.zeros(P * n_teams)] if theta0 is None else theta0.copy()
    strat = [(np.full(g[0].shape[0], 1 / g[0].shape[0]), np.full(g[3].shape[0], 1 / g[3].shape[0])) for g, _ in games]
    hist = []
    for _ in range(n_iter):
        data = _pseudo_data(games, strat, choices)
        bounds = None
        if fix_nu:                                  # H_winmax: nu = 1 for every team (log nu = 0 globally and in deviations)
            bounds = [(None, None)] * len(flat)
            for j in [1] + [P + P * i + 1 for i in range(n_teams)]:
                bounds[j] = (0.0, 0.0)
            flat = np.clip(flat, [b_[0] if b_[0] is not None else -np.inf for b_ in bounds],
                           [b_[1] if b_[1] is not None else np.inf for b_ in bounds])
        r = optimize.minimize(_nll_grad, flat, args=(data, n_teams), jac=True, method="L-BFGS-B", bounds=bounds)
        p = _probs(_team_params(r.x, n_teams), data)[2]
        strat = [(p[2 * c, :g[0].shape[0]], p[2 * c + 1, :g[3].shape[0]]) for c, (g, _) in enumerate(games)]
        change = float(np.abs(r.x - flat).max())
        flat = r.x
        hist.append((float(r.fun), change))
        if change < tol:
            break
    t = _team_params(flat, n_teams)
    out = np.column_stack([np.exp(t[:, 0]), np.exp(t[:, 1]), t[:, 2:]])
    return out, strat, hist
