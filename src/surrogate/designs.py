"""v0.2 designs (2 Nov 2026; V02_PLAN sections 4.1, 4.2, 8): the held-out grid T, candidate pools P, and the four seeded
selection orders compared in the pre-registered test (active learning, random, nested regular grid, space-filling).

Coordinates: chi in degrees (-30..30), sigma in units of s (-1..1); distances use the scaled x = (chi / 30, sigma).
T: 7 x 7, chi in {-30, -20, ..., 30}, sigma in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}.
Pool P of about M points (make_pool): a regular part n_chi x n_sig at cell centres, chi_i = -30 + 60 (i + 0.5) / n_chi,
sig_j = -1 + (2 j + 1) / n_sig, plus chi = -30 and +30 at every sig_j (edge points, Q6b). n_sig is even, so no sig_j equals a
T level (k / 3): P is disjoint from T. (n_chi, n_sig) minimise |(n_chi + 2) n_sig - M|, then |n_chi - n_sig|.
Orders (lists of pool indices, nested: the first n of an order are the n-point design):
  random_order: seeded permutation.
  spacefill_order: greedy maximin (farthest point first) in scaled x, starting from a seeded random point.
  grid_designs: nested regular sub-grids of the regular part (Q7): 1D nested index orders (ends, then repeated bisection)
      on each axis; sub-grid (a, b) = first a chi indices x first b sig indices; starts at (2, 2) and grows the axis with the
      smaller filled fraction (chi first on a tie) up to n_chi x n_sig. Each design is a superset of the previous one.
  active learning: implemented in scripts/v02_pool_study.py (seeded random start = the first N_INIT of random_order).
"""
import numpy as np

T_CHI = np.arange(-30, 31, 10, dtype=float)
T_SIG = np.array([-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1])


def grid_T():
    c, s = np.meshgrid(T_CHI, T_SIG, indexing="ij")
    return np.column_stack([c.ravel(), s.ravel()])


def pool_shape(M):
    best = None
    for ns in range(2, 400, 2):
        for nc in range(2, 400):
            key = (abs((nc + 2) * ns - M), abs(nc - ns))
            if best is None or key < best[0]:
                best = (key, nc, ns)
    return best[1], best[2]


def make_pool(M):
    """Returns (points (n, 2) [chi, sig], n_chi, n_sig); rows 0 .. n_chi*n_sig-1 are the regular part (chi-major)."""
    nc, ns = pool_shape(M)
    chi = -30 + 60 * (np.arange(nc) + 0.5) / nc
    sig = -1 + (2 * np.arange(ns) + 1) / ns
    c, s = np.meshgrid(chi, sig, indexing="ij")
    reg = np.column_stack([c.ravel(), s.ravel()])
    edge = np.array([[e, v] for e in (-30.0, 30.0) for v in sig])
    return np.vstack([reg, edge]), nc, ns


def scaled(P):
    P = np.asarray(P, float)
    return np.column_stack([P[:, 0] / 30.0, P[:, 1]])


def random_order(n, seed):
    return list(np.random.default_rng(seed).permutation(n))


def spacefill_order(P, seed):
    X = scaled(P)
    first = int(np.random.default_rng(seed).integers(len(X)))
    order, d = [first], np.linalg.norm(X - X[first], axis=1)
    d[first] = -1
    for _ in range(len(X) - 1):
        nxt = int(np.argmax(d))
        order.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1)); d[order] = -1
    return order


def nested_1d(n):
    """Index order: 0, n-1, then midpoints of the largest gaps (breadth first), every index exactly once."""
    order, gaps = [0, n - 1] if n > 1 else [0], [(0, n - 1)]
    while len(order) < n:
        nxt = []
        for a, b in gaps:
            if b - a > 1:
                m = (a + b) // 2
                order.append(m); nxt += [(a, m), (m, b)]
        gaps = nxt
    return order


def grid_designs(n_chi, n_sig):
    """List of nested sub-grids, each a list of regular-part pool indices (chi-major: index = i * n_sig + j)."""
    oc, os_ = nested_1d(n_chi), nested_1d(n_sig)
    a, b, out = 2, 2, []
    while True:
        out.append([i * n_sig + j for i in oc[:a] for j in os_[:b]])
        if a == n_chi and b == n_sig:
            return out
        if (a / n_chi <= b / n_sig and a < n_chi) or b == n_sig:
            a += 1
        else:
            b += 1
