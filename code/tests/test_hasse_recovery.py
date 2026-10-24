"""Parameter-recovery check for lolsetup.hasse on synthetic data drawn from the model itself.

Run: python -m tests.test_hasse_recovery   (light: ~600 setups x 3 snapshots, K = 3)
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score

from lolsetup import hasse, lattice

rng = np.random.default_rng(7)
P5, FULL = hasse.P5, hasse.FULL


def lab(p):
    return "|".join("".join("TJMBS"[r] for r in b) for b in p)


def idx(label):
    blocks = tuple(tuple("TJMBS".index(ch) for ch in blk) for blk in label.split("|"))
    return hasse.P5_INDEX[lattice.canon(blocks)]


TRUE = [  # centre paths for T = 3 (s-60, s, s+60)
    [idx("BS|J|M|T"), idx("JMS|B|T"), idx("JMBS|T")],     # gather to 4-1 with top split
    [idx("BS|JM|T"), idx("JMBS|T"), idx("TJMBS")],        # full 5-man stack
    [idx("T|J|M|B|S"), idx("BS|J|M|T"), idx("BS|J|M|T")],  # stay spread (concede / farm)
]
THETA, IOTA = [2.5, 3.0, 2.0], [0.5, 0.8, 0.3]
PIT_PREF = [  # zone preference: pattern 0 and 1 go to the pit (zone 0), pattern 2 stays in lanes (zone 3/5)
    {0: 1.2, 3: 0.2}, {0: 1.8}, {3: 1.0, 5: 1.0},
]


def sample_setup(c_seq, cls):
    snaps, prev = [], None
    for t, c in enumerate(c_seq):
        A = tuple(r for r in FULL if not (rng.random() < 0.1 and r == rng.integers(0, 5)))
        PA, _, D, R = hasse.alive_table(A)
        if prev is None:
            inert = np.zeros(len(PA))
        else:
            C = tuple(sorted(set(A) & set(prev[0])))
            inert = hasse.inertia_vector(A, C, lattice.restrict(prev[1], C))
        logits = -THETA[c] * D[:, R[TRUE[c][t]]] - IOTA[c] * inert
        p = np.exp(logits - logsumexp(logits))
        pi = PA[rng.choice(len(PA), p=p)]
        zones = {}
        for B in pi:
            u = np.zeros(hasse.NZ)
            for z, v in PIT_PREF[c].items():
                u[z] += v * len(B)
            z = rng.choice(hasse.NZ, p=np.exp(u - logsumexp(u)))
            for r in B:
                zones[r] = int(z)
        snaps.append({"A": A, "partition": pi, "zones": zones, "s": t / 2})
        prev = (A, pi)
    return hasse.make_setup(snaps, cls, group=f"g{rng.integers(0, 60)}")


def main():
    data, truth = [], []
    for i in range(600):
        c = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        seq = [c, c, c] if rng.random() > 0.1 else [c, c, int(rng.integers(0, 3))]
        data.append(sample_setup(seq, cls=int(rng.integers(0, 3))))
        truth.append(c)
    B = hasse.Batch(data)
    best = None
    for seed in range(5):
        m = hasse.Hasse(K=3, n_classes=3, T=3, seed=seed).fit(B, max_iter=100)
        print(f"restart {seed}: loglik {m.loglik_:.1f} after {m.n_iter_} iterations")
        if best is None or m.loglik_ > best.loglik_:
            best = m
    _, g, _ = best.forward_backward(B)
    est = [int(np.argmax(x[0])) for x in g]
    print("ARI(first-snapshot assignment vs truth):", round(adjusted_rand_score(truth, est), 3))
    print("true centre paths:", [[lab(P5[m]) for m in path] for path in TRUE])
    for s in best.summary():
        print(s)



def main_missing(share=0.4):
    """Risk-set variant: the last snapshot is missing (objective already taken) for ~share of setups."""
    data, truth = [], []
    for _ in range(600):
        c = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        e = sample_setup([c, c, c], cls=int(rng.integers(0, 3)))
        if rng.random() < share:
            e.obs[-1] = hasse.Obs((), 0, np.zeros(1), 0.0, [], e.obs[-1].s, missing=True)
        data.append(e)
        truth.append(c)
    B = hasse.Batch(data)
    best = max((hasse.Hasse(K=3, n_classes=3, T=3, seed=s).fit(B, max_iter=100) for s in range(3)),
               key=lambda m: m.loglik_)
    _, g, _ = best.forward_backward(B)
    print("missing-last-snapshot variant, ARI:", round(adjusted_rand_score(truth, g[:, 0].argmax(1)), 3))
    for s in best.summary():
        print(s)


def main_init(Ks=(3, 5), restarts=6):
    """Restart agreement: random centre paths vs k-medoids initialisation on the lattice path distance."""
    data = []
    for _ in range(600):
        c = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        data.append(sample_setup([c, c, c], cls=int(rng.integers(0, 3))))
    B = hasse.Batch(data)
    for K in Ks:
        for name in ("random", "medoid"):
            ll = []
            for s in range(restarts):
                init = hasse.medoid_init(data, K, seed=s)[0] if name == "medoid" else None
                ll.append(hasse.Hasse(K=K, n_classes=3, T=3, seed=s, init_mu=init).fit(B, max_iter=100).loglik_)
            ll = np.array(ll)
            print(f"K={K} {name:6s}: best {ll.max():.1f}  spread(best-worst) {ll.max() - ll.min():.1f}  "
                  f"share within 1 of best {(ll > ll.max() - 1).mean():.2f}")


if __name__ == "__main__":
    main()
    main_missing()
    main_init()
