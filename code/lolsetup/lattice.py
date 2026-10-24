"""The partition lattice Pi(A) of the alive roles A (|A| <= 5): enumeration, join, restriction, metric.

A partition is a tuple of sorted blocks (tuples of role indices 0..4), blocks ordered by
(-size, first element). Refinement order: sigma <= tau if every block of sigma lies inside a block
of tau (sigma is finer). Rank r(pi) = |A| - |pi|.

Join metric  d(sigma, tau) = |sigma| + |tau| - 2 |sigma v tau|  = 2 r(sigma v tau) - r(sigma) - r(tau).
Pi(A) is upper semimodular, so d is a metric; on every shortest Hasse path from sigma to tau there
are exactly |sigma| - |sigma v tau| merges and |tau| - |sigma v tau| splits. self_test() checks the
metric against breadth-first-search distances on the Hasse (covering) graph of Pi_5.
"""
from functools import lru_cache
from itertools import combinations


def canon(blocks):
    return tuple(sorted((tuple(sorted(b)) for b in blocks if b), key=lambda b: (-len(b), b)))


@lru_cache(maxsize=None)
def partitions(ground):
    """All set partitions of the tuple `ground` (restricted growth strings)."""
    ground = tuple(ground)
    if not ground:
        return ((),)
    first, rest = ground[0], ground[1:]
    out = []
    for p in partitions(rest):
        out.append(canon(((first,),) + p))
        for i in range(len(p)):
            out.append(canon(p[:i] + ((first,) + p[i],) + p[i + 1:]))
    return tuple(sorted(set(out)))


def join(a, b):
    """Finest common coarsening (blocks that share an element are merged transitively)."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for blk in a + b:
        for x in blk:
            parent.setdefault(x, x)
        for x in blk[1:]:
            parent[find(x)] = find(blk[0])
    groups = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return canon(groups.values())


def restrict(p, keep):
    keep = set(keep)
    return canon(tuple(x for x in b if x in keep) for b in p)


def dist(a, b):
    return len(a) + len(b) - 2 * len(join(a, b))


def covers(p):
    """Partitions covering p (one merge of two blocks)."""
    return [canon(tuple(b for k, b in enumerate(p) if k not in (i, j)) + (p[i] + p[j],))
            for i, j in combinations(range(len(p)), 2)]


def self_test():
    from collections import deque
    P = partitions((0, 1, 2, 3, 4))
    assert len(P) == 52, len(P)
    edges = {(p, q) for p in P for q in covers(p)}
    assert len(edges) == 160, len(edges)
    finest = canon(tuple((i,) for i in range(5)))
    spectrum = [sum(1 for p in P if dist(finest, p) == k) for k in range(5)]
    assert spectrum == [1, 10, 25, 15, 1], spectrum
    shapes = {}
    for p in P:
        shapes.setdefault(tuple(len(b) for b in p), 0)
        shapes[tuple(len(b) for b in p)] += 1
    assert sorted(shapes.values()) == [1, 1, 5, 10, 10, 10, 15], shapes
    adj = {p: set() for p in P}
    for p, q in edges:
        adj[p].add(q); adj[q].add(p)
    worst = 0
    for s in P:  # BFS distances on the Hasse graph equal the join metric for every pair
        seen = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    dq.append(v)
        for t in P:
            assert seen[t] == dist(s, t), (s, t, seen[t], dist(s, t))
            worst = max(worst, seen[t])
    assert worst == 4
    for a in P:  # triangle inequality
        for b in P:
            for c in P:
                assert dist(a, c) <= dist(a, b) + dist(b, c)
    return {"partitions": len(P), "cover_edges": len(edges), "spectrum_from_finest": spectrum, "diameter": worst}


if __name__ == "__main__":
    print(self_test())
