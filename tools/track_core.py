"""Core tracker that follows the core (Ansh's decision, Session 8): excess-energy-weighted centroid in a circular window
in the (y, z) projection, centred on the previous core position and re-centred on the new centroid until it stops moving.
Seed: z from the periodic (circular) mean over a band |y - y_prev| < R along the whole cell (as the old tracker, but centred on
the previous core height), so a core that ran several jumps within a step is still found; y seed = y_prev.

Usage: python track_core.py <pe dump from in.pe_dump> <y_prev> <z_prev> <Lz> [--R 6.0]
Prints: "CORE y=<A> z=<A> iterations=<n> n_atoms=<n>" (z wrapped into [0, Lz)).
Weights: excess = pe - (-4.1224) eV for excess > 0.01 eV (same zero and threshold as the old tracker in in.step).
Differences from the old tracker: the window is a circle of radius R around the current estimate (the old one was a band
|y - y0| < 4 A around the STARTING glide plane, spanning the whole z length), and it is iterated, so it follows a core that
moves in y as well as z. A circle is compatible with the 3-fold core symmetry, so it does not bias y.
"""
import argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("y_prev", type=float); ap.add_argument("z_prev", type=float); ap.add_argument("Lz", type=float)
ap.add_argument("--R", type=float, default=6.0)
a = ap.parse_args()
L = open(a.dump).read().splitlines()
i = L.index(next(l for l in L if l.startswith("ITEM: ATOMS")))
d = np.loadtxt(L[i + 1:])
y, z, ex = d[:, 2], d[:, 3], d[:, 4] + 4.1224
w = np.where(ex > 0.01, ex, 0.0)
band = (np.abs(y - a.y_prev) < a.R) & (w > 0)
ph = 2 * np.pi * z[band] / a.Lz
yc, zc = a.y_prev, np.arctan2((w[band] * np.sin(ph)).sum(), (w[band] * np.cos(ph)).sum()) * a.Lz / (2 * np.pi)
for it in range(1, 101):
    dz = (z - zc + a.Lz / 2) % a.Lz - a.Lz / 2                # minimum image along periodic z
    m = (np.hypot(y - yc, dz) < a.R) & (w > 0)
    yn, zn = (w[m] * y[m]).sum() / w[m].sum(), zc + (w[m] * dz[m]).sum() / w[m].sum()
    done = np.hypot(yn - yc, zn - zc) < 1e-5
    yc, zc = yn, zn
    if done:
        break
print(f"CORE y={yc:.4f} z={zc % a.Lz:.4f} iterations={it} n_atoms={m.sum()}")
