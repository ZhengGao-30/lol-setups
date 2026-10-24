"""HASSE: lattice-structured sticky HMM for objective setups (discovery layer).

A team-setup e is a short sequence t = 1..T of observations y_et = (A_t, pi_t, {(B, g_B)}):
alive roles A_t, formation pi_t in Pi(A_t), and the battlefield zone g_B of each block B.
Pattern c has a centre path mu_c,1:T in Pi_5, rigidity theta_c, inertia iota_c and zone
preferences psi_c[r, h] = a_c[r, h] + b_c[r, h] * s_t (role r, zone h, time covariate s_t):

  P_c(pi_t = sigma | pi_{t-1}, A_t) = exp(-theta_c d(sigma, mu_c,t|A_t) - iota_c d(sigma|C, pi_{t-1}|C)) / Z
  P_c(g_t | pi_t)                  = prod_{B in pi_t} softmax_h( sum_{r in B} psi_c[r, h] )[g_B]
  z_1 ~ Cat(w_class(e)),  P(z_t = c' | z_{t-1} = c) = (1 - xi) 1[c' = c] + xi q[c, c']

d is the join metric of the partition lattice (lolsetup.lattice); C = A_t & A_{t-1}; dead roles are
handled by restriction, never imputed. Context variables never enter. Fitted by EM: exact
forward-backward over K states, exact Viterbi over the 52 centre partitions x T for mu, a concave
2-D maximisation with analytic gradient for (theta, iota) (exponential family: gradient = observed
minus expected distances), penalised multinomial logit for psi, closed forms for w, xi, q.
All likelihood terms are vectorised over observations grouped by alive set.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import lattice

FULL = (0, 1, 2, 3, 4)
P5 = lattice.partitions(FULL)
P5_INDEX = {p: i for i, p in enumerate(P5)}
NZ = 7  # zones: pit, own_pitside, opp_pitside, pit_lane, mid, far, base (base is the reference)
CHI = 1.0  # centre-path smoothness penalty (Hasse steps between consecutive centres)
PEN_A, PEN_B = 0.1, 0.1  # Gaussian penalties on zone parameters
BOUNDS = (0.0, 20.0)


@lru_cache(maxsize=None)
def alive_table(A):
    """For alive set A (tuple): partitions of A, index, distance matrix, restriction map Pi_5 -> Pi(A)."""
    PA = lattice.partitions(A)
    idx = {p: i for i, p in enumerate(PA)}
    D = np.array([[lattice.dist(p, q) for q in PA] for p in PA], dtype=float)
    R = np.array([idx[lattice.restrict(mu, A)] for mu in P5])
    return PA, idx, D, R


@lru_cache(maxsize=None)
def inertia_vector(A, C, prev_restricted):
    """d(sigma|C, prev|C) for every sigma in Pi(A)."""
    PA = alive_table(A)[0]
    return np.array([lattice.dist(lattice.restrict(s, C), prev_restricted) for s in PA], dtype=float)


@lru_cache(maxsize=None)
def refine_matrix():
    """R[i, j] = True iff partition i refines partition j (j is a coarsening of i): the allowed centre steps of H_mono."""
    return np.array([[lattice.join(a, b) == b for b in P5] for a in P5])


@dataclass
class Obs:
    A: tuple
    pi_idx: int
    inertia: np.ndarray        # over Pi(A); zeros at t = 1
    inertia_obs: float
    blocks: list               # list of (role tuple, zone index)
    s: float                   # time covariate in [0, 1]
    missing: bool = False      # e.g. objective already taken before this post-spawn snapshot (risk set)


@dataclass
class Setup:
    obs: list
    cls: int                   # objective class index
    group: str                 # series id for grouped CV
    meta: dict = field(default_factory=dict)


def make_setup(snapshots, cls, group, meta=None):
    """snapshots: list of dicts with keys A (alive role idx), partition (blocks), s, and either
    block_zones (zone idx per block, in canonical block order) or zones (role -> zone idx; block zone =
    most common member zone). A snapshot with missing=True contributes no likelihood term."""
    obs, prev = [], None
    for snap in snapshots:
        if snap.get("missing"):
            obs.append(Obs((), 0, np.zeros(1), 0.0, [], float(snap["s"]), missing=True))
            continue
        A = tuple(sorted(snap["A"]))
        PA, idx, _, _ = alive_table(A)
        pi = lattice.canon(snap["partition"])
        if prev is None:
            inert, inert_obs = np.zeros(len(PA)), 0.0
        else:
            C = tuple(sorted(set(A) & set(prev[0])))
            prev_r = lattice.restrict(prev[1], C)
            inert = inertia_vector(A, C, prev_r)
            inert_obs = float(lattice.dist(lattice.restrict(pi, C), prev_r))
        if "block_zones" in snap:
            blocks = list(zip(pi, snap["block_zones"]))
        else:
            blocks = []
            for B in pi:
                zs = [snap["zones"][r] for r in B]
                blocks.append((B, min(set(zs), key=lambda z: (-zs.count(z), z))))
        obs.append(Obs(A, idx[pi], inert, inert_obs, blocks, float(snap["s"])))
        prev = (A, pi)
    return Setup(obs, cls, group, meta or {})


class Batch:
    """Flat, grouped arrays for vectorised likelihoods. All setups must have the same T."""

    def __init__(self, data):
        self.data = data
        self.T = len(data[0].obs)
        assert all(len(e.obs) == self.T for e in data), "all setups must have the same number of snapshots"
        self.E = len(data)
        flat = [(i, t, o) for i, e in enumerate(data) for t, o in enumerate(e.obs)]
        self.N = len(flat)
        self.obs_e = np.array([i for i, _, _ in flat]); self.obs_t = np.array([t for _, t, _ in flat])
        self.cls = np.array([e.cls for e in data])
        self.groups = []
        by_A = {}
        for n, (_, _, o) in enumerate(flat):
            if not o.missing:
                by_A.setdefault(o.A, []).append(n)
        for A, ns in by_A.items():
            _, _, D, R = alive_table(A)
            ns = np.array(ns)
            self.groups.append({
                "n": ns, "Dm": D[:, R],                                         # |PA| x 52
                "inert": np.stack([flat[k][2].inertia for k in ns]),            # n x |PA|
                "pi": np.array([flat[k][2].pi_idx for k in ns]),
                "inert_obs": np.array([flat[k][2].inertia_obs for k in ns]),
            })
        bo, bm, bs, bz = [], [], [], []
        for n, (_, _, o) in enumerate(flat):
            for B, z in o.blocks:
                m = np.zeros(5); m[list(B)] = 1
                bo.append(n); bm.append(m); bs.append(o.s); bz.append(z)
        self.b_obs, self.b_mask = np.array(bo), np.array(bm)
        self.b_s, self.b_zone = np.array(bs), np.array(bz)

    def part_logp_all(self, theta, iota):
        """N x 52: log P(pi_obs | ...) for every candidate centre mu (0 for missing snapshots)."""
        out = np.zeros((self.N, len(P5)))
        for g in self.groups:
            logits = -theta * g["Dm"][None, :, :] - iota * g["inert"][:, :, None]   # n x |PA| x 52
            out[g["n"]] = (-theta * g["Dm"][g["pi"]] - iota * g["inert_obs"][:, None]) - logsumexp(logits, axis=1)
        return out

    def zone_logp(self, a, b):
        """N-vector: sum over blocks of log softmax zone probability under (a, b)."""
        U = self.b_mask @ a + (self.b_mask * self.b_s[:, None]) @ b
        ll = U[np.arange(len(U)), self.b_zone] - logsumexp(U, axis=1)
        out = np.zeros(self.N)
        np.add.at(out, self.b_obs, ll)
        return out


def observed_paths(data):
    """Per setup: list over t of the observed partition (tuple of blocks over the alive roles) or None."""
    out = []
    for e in data:
        out.append([None if o.missing else alive_table(o.A)[0][o.pi_idx] for o in e.obs])
    return out


def path_distance(pa, pb):
    """Mean lattice distance over co-observed snapshots after restriction to the common alive roles
    (1e9-free: setups with no co-observed snapshot get the maximum distance 4)."""
    ds = []
    for x, y in zip(pa, pb):
        if x is None or y is None:
            continue
        keep = tuple(sorted(set(r for b in x for r in b) & set(r for b in y for r in b)))
        ds.append(lattice.dist(lattice.restrict(x, keep), lattice.restrict(y, keep)))
    return float(np.mean(ds)) if ds else 4.0


def medoid_init(data, K, seed=0, n_sub=400):
    """k-medoids (k-medoids++ seeding + alternating updates) on the partition-lattice path distance over a
    subsample of observed snapshot paths; returns K centre paths (K x T indices into Pi_5) and the medoid ids.
    A centre path takes the medoid's observed partition at each t (missing -> previous, then next observed;
    dead roles are singletons)."""
    rng = np.random.default_rng(seed)
    paths = observed_paths(data)
    T = len(paths[0])
    pool = [i for i, p in enumerate(paths) if sum(x is not None for x in p) >= max(2, T // 3)]
    if len(pool) < K:
        return None
    sub = list(rng.choice(pool, size=min(n_sub, len(pool)), replace=False))
    n = len(sub)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = path_distance(paths[sub[i]], paths[sub[j]])
    med = [int(rng.integers(n))]
    while len(med) < K:
        dmin = D[:, med].min(1) ** 2
        med.append(int(rng.choice(n, p=dmin / dmin.sum())) if dmin.sum() > 0 else int(rng.integers(n)))
    for _ in range(20):
        lab = D[:, med].argmin(1)
        new = []
        for k in range(K):
            mem = np.where(lab == k)[0]
            new.append(int(mem[D[np.ix_(mem, mem)].sum(1).argmin()]) if len(mem) else med[k])
        if new == med:
            break
        med = new
    mu = np.zeros((K, T), dtype=int)
    for k, m in enumerate(med):
        p = paths[sub[m]]
        last = next(x for x in p if x is not None)
        for t in range(T):
            if p[t] is not None:
                last = p[t]
            full = set(r for b in last for r in b)
            blocks = tuple(last) + tuple((r,) for r in FULL if r not in full)
            mu[k, t] = P5_INDEX[lattice.canon(blocks)]
    return mu, [sub[m] for m in med]


class Hasse:
    def __init__(self, K, n_classes, T, seed=0, init_mu=None, mono=False):
        self.K, self.C, self.T = K, n_classes, T
        self.mono = mono  # H_mono: centre paths may only merge (gather), never split
        self.rng = np.random.default_rng(seed)
        self.mu = self.rng.integers(0, len(P5), size=(K, T)) if init_mu is None else np.array(init_mu)
        self.theta = np.full(K, 1.0)
        self.iota = np.full(K, 0.5)
        self.a = self.rng.normal(0, 0.3, size=(K, 5, NZ)); self.a[:, :, -1] = 0
        self.b = np.zeros((K, 5, NZ))
        self.w = np.full((n_classes, K), 1.0 / K)
        self.xi = 0.05
        self.q = np.full((K, K), 1.0 / max(K - 1, 1)); np.fill_diagonal(self.q, 0)

    def emissions(self, B):
        """E x T x K emission log-likelihoods."""
        L = np.zeros((B.E, B.T, self.K))
        for c in range(self.K):
            lp = B.part_logp_all(self.theta[c], self.iota[c])
            v = lp[np.arange(B.N), self.mu[c, B.obs_t]] + B.zone_logp(self.a[c], self.b[c])
            L[B.obs_e, B.obs_t, c] = v
        return L

    def forward_backward(self, B, L=None):
        L = self.emissions(B) if L is None else L
        logA = np.log(np.maximum((1 - self.xi) * np.eye(self.K) + self.xi * self.q, 1e-300))
        la = np.zeros_like(L); lb = np.zeros_like(L)
        la[:, 0] = np.log(np.maximum(self.w[B.cls], 1e-300)) + L[:, 0]
        for t in range(1, B.T):
            la[:, t] = logsumexp(la[:, t - 1, :, None] + logA[None], axis=1) + L[:, t]
        for t in range(B.T - 2, -1, -1):
            lb[:, t] = logsumexp(logA[None] + (L[:, t + 1] + lb[:, t + 1])[:, None, :], axis=2)
        lz = logsumexp(la[:, -1], axis=1)
        gamma = np.exp(la + lb - lz[:, None, None])
        xi = np.exp(la[:, :-1, :, None] + logA[None, None] + (L[:, 1:] + lb[:, 1:])[:, :, None, :] - lz[:, None, None, None])
        return lz, gamma, xi

    def _update_mu(self, B, gamma, c):
        lp = B.part_logp_all(self.theta[c], self.iota[c])            # N x 52
        S = np.zeros((B.T, len(P5)))
        np.add.at(S, B.obs_t, gamma[B.obs_e, B.obs_t, c][:, None] * lp)
        Dfull = alive_table(FULL)[2]
        best = S[0].copy(); back = np.zeros((B.T, len(P5)), dtype=int)
        for t in range(1, B.T):
            cand = best[:, None] - CHI * Dfull
            if self.mono:
                cand = np.where(refine_matrix(), cand, -np.inf)   # row = previous centre, column = next centre
            back[t] = np.argmax(cand, axis=0)
            best = cand[back[t], np.arange(len(P5))] + S[t]
        path = [int(np.argmax(best))]
        for t in range(B.T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        self.mu[c] = path[::-1]

    def _update_theta_iota(self, B, gamma, c):
        wts = gamma[B.obs_e, B.obs_t, c]
        sel = self.mu[c, B.obs_t]

        def f(x):
            th, io_ = x
            val, g_th, g_io = 0.0, 0.0, 0.0
            for g in B.groups:
                w = wts[g["n"]]
                Dsel = g["Dm"][:, sel[g["n"]]].T                       # n x |PA|
                logits = -th * Dsel - io_ * g["inert"]
                lse = logsumexp(logits, axis=1)
                p = np.exp(logits - lse[:, None])
                d1 = Dsel[np.arange(len(w)), g["pi"]]
                val += w @ (th * d1 + io_ * g["inert_obs"] + lse)
                g_th += w @ (d1 - (p * Dsel).sum(1))
                g_io += w @ (g["inert_obs"] - (p * g["inert"]).sum(1))
            return val, np.array([g_th, g_io])

        res = minimize(f, [self.theta[c], self.iota[c]], jac=True, method="L-BFGS-B", bounds=[BOUNDS, BOUNDS])
        self.theta[c], self.iota[c] = res.x

    def _update_zones(self, B, gamma, c):
        W = gamma[B.obs_e[B.b_obs], B.obs_t[B.b_obs], c]
        M, S, Z = B.b_mask, B.b_s, B.b_zone
        nfree = NZ - 1
        Y = np.eye(NZ)[Z]

        def f(x):
            a = np.concatenate([x[:5 * nfree].reshape(5, nfree), np.zeros((5, 1))], 1)
            b = np.concatenate([x[5 * nfree:].reshape(5, nfree), np.zeros((5, 1))], 1)
            U = M @ a + (M * S[:, None]) @ b
            lse = logsumexp(U, axis=1)
            ll = W @ (U[np.arange(len(Z)), Z] - lse)
            G = W[:, None] * (Y - np.exp(U - lse[:, None]))
            ga = (M.T @ G)[:, :nfree]; gb = ((M * S[:, None]).T @ G)[:, :nfree]
            val = -ll + PEN_A * (a ** 2).sum() + PEN_B * (b ** 2).sum()
            grad = np.concatenate([(-ga + 2 * PEN_A * a[:, :nfree]).ravel(), (-gb + 2 * PEN_B * b[:, :nfree]).ravel()])
            return val, grad

        x0 = np.concatenate([self.a[c][:, :nfree].ravel(), self.b[c][:, :nfree].ravel()])
        res = minimize(f, x0, jac=True, method="L-BFGS-B")
        self.a[c][:, :nfree] = res.x[:5 * nfree].reshape(5, nfree)
        self.b[c][:, :nfree] = res.x[5 * nfree:].reshape(5, nfree)

    def m_step(self, B, gamma, xi):
        cnt = np.full((self.C, self.K), 1e-3)
        np.add.at(cnt, B.cls, gamma[:, 0])
        self.w = cnt / cnt.sum(1, keepdims=True)
        tot = xi.sum((0, 1))
        stay = np.trace(tot) + 1e-6
        off = tot.copy(); np.fill_diagonal(off, 0)
        self.xi = float(np.clip(off.sum() / (off.sum() + stay), 1e-4, 0.5))
        if self.K > 1:
            off = off + 1e-6; np.fill_diagonal(off, 0)
            self.q = off / off.sum(1, keepdims=True)
        for c in range(self.K):
            self._update_mu(B, gamma, c)
            self._update_theta_iota(B, gamma, c)
            self._update_zones(B, gamma, c)

    def fit(self, data, max_iter=200, tol=1e-7, verbose=False):
        B = data if isinstance(data, Batch) else Batch(data)
        prev = -np.inf
        for it in range(max_iter):
            lz, gamma, xi = self.forward_backward(B)
            ll = float(lz.sum())
            self.m_step(B, gamma, xi)
            if verbose:
                print(it, round(ll, 3))
            if abs(ll - prev) < tol * abs(ll):
                break
            prev = ll
        self.n_iter_ = it + 1
        lz, gamma, _ = self.forward_backward(B)
        self.loglik_ = float(lz.sum())
        self.gamma_ = gamma
        return self

    def loglik_per_setup(self, data):
        B = data if isinstance(data, Batch) else Batch(data)
        return self.forward_backward(B)[0]

    def n_params(self):
        return self.K * (self.T + 2 + 2 * 5 * (NZ - 1)) + self.C * (self.K - 1) + 1 + self.K * max(self.K - 2, 0)

    def summary(self):
        return [{"pattern": c, "weight_by_class": self.w[:, c].round(3).tolist(),
                 "centre_path": [" | ".join("".join("TJMBS"[r] for r in b) for b in P5[m]) for m in self.mu[c]],
                 "theta": round(float(self.theta[c]), 3), "iota": round(float(self.iota[c]), 3)} for c in range(self.K)]
