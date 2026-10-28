"""Differential-displacement (DD) map of a screw core from a LAMMPS data file (e.g. one Peierls-ramp step).

Usage (from a sim folder): SCREW_NZ=<nz> python ../../tools/ddmap_data.py <state.data> <out.png> [title]
Needs screw_down.data.ref.npy / .core.npy (perfect positions, inserted core) in the current folder.
Same method and style as sims/01_screw_cell/analyze.py (plus: extra box shear of chi != 0 steps removed, see below): u_x relative to the perfect crystal, mod b; arrow on each
pair of neighbouring [111] columns, length |DD| / (b/3) of the column spacing.
Burgers triangle = the triangle of three mutually neighbouring columns (within 12 A of the inserted core) with the
largest sum of |DD| (its wrapped arrows add up to b: it holds the Burgers vector); printed with its three |DD|,
centroid and orientation. That alone does NOT show the core shape: the script also prints the sum of |DD| on the
6 bonds of each column. Compact easy core: the 3 triangle columns share it equally (~1.10 b each at sigma = 0);
a core centred on one column shows one column well above the others.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sims", "01_screw_cell"))
from build import A, B, LZ

path, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else path
ref = np.load("screw_down.data.ref.npy")
y0, z0 = np.load("screw_down.data.core.npy")
XZ = B / 2

L = open(path).read().splitlines()
n_atoms = int(next(l for l in L if l.strip().endswith("atoms")).split()[0])
i0 = L.index("Atoms # atomic") + 2
d = np.array([l.split()[:5] for l in L[i0:i0 + n_atoms]], float)
d = d[d[:, 0].argsort()]
assert len(d) == len(ref)
x, y, z = d[:, 2].copy(), d[:, 3], d[:, 4]
# Box of this state: chi != 0 steps (Session 10) carry an extra box shear gamma_xz on top of the PAD tilt b/2 (and a small
# eps_zz). Atoms wrapped across z pick up the actual tilt; the uniform extra shear (xz - b/2) * z / Lz is removed from u_x.
XZcur = float(next(l for l in L[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
zl = next(l for l in L[:30] if l.rstrip().endswith("zlo zhi")).split()
LZcur = float(zl[1]) - float(zl[0])
x -= np.round((z - ref[:, 2]) / LZ) * XZcur
x -= (XZcur - XZ) * z / LZcur
ux = (x - ref[:, 0] + B / 2) % B - B / 2           # x wraps by LX = 6b, which drops out mod b

col = (ref[:, 0] >= -1e-6) & (ref[:, 0] < B - 1e-6)
P = ref[col][:, 1:] - [y0, z0]                     # (y, z) of each column relative to the inserted core
h = np.round(ref[col][:, 0] / (B / 3)).astype(int) % 3
u = ux[col]
idx = np.where(np.hypot(*P.T) < 16)[0]
dnn = A * np.sqrt(2 / 3)
dd = {}
for a_ in range(len(idx)):
    for b_ in range(a_ + 1, len(idx)):
        i, j = idx[a_], idx[b_]
        if np.linalg.norm(P[i] - P[j]) < dnn + 0.1:
            du = (u[j] - u[i] + B / 2) % B - B / 2
            dd[(i, j)], dd[(j, i)] = du, -du

# core triangle: largest total |DD| over all neighbour triangles near the inserted core
best = None
near = [i for i in idx if np.hypot(*P[i]) < 12]
for i in near:
    for j in near:
        for k in near:
            if i < j < k and (i, j) in dd and (j, k) in dd and (i, k) in dd:
                s = abs(dd[(i, j)]) + abs(dd[(j, k)]) + abs(dd[(i, k)])
                if best is None or s > best[0]:
                    best = (s, (i, j, k))
s, (i, j, k) = best
tri = [abs(dd[(i, j)]) / B, abs(dd[(j, k)]) / B, abs(dd[(i, k)]) / B]
cen = P[[i, j, k]].mean(0)
ys = sorted(P[[i, j, k], 0])
orient = "up" if abs(ys[1] - ys[0]) < 1e-3 else "down"
dcol = np.sort(np.hypot(*(P[idx] - cen).T))[:3]
print(f"{path}: Burgers triangle |DD|/b = {tri[0]:.4f}, {tri[1]:.4f}, {tri[2]:.4f} (sum {s / B:.4f} b; 1/3 = 0.3333 each)")
print(f"{path}: triangle centroid relative to inserted core: dy = {cen[0]:+.4f} A, dz = {cen[1]:+.4f} A "
      f"(= {cen[1] / (A * np.sqrt(6) / 3):+.3f} jumps); orientation '{orient}' (easy core was 'down')")
print(f"{path}: distance from centroid to the 3 nearest columns: {dcol[0]:.4f}, {dcol[1]:.4f}, {dcol[2]:.4f} A "
      f"(triangle centre: a*sqrt2/3 = {A * np.sqrt(2) / 3:.4f} A to all three)")
# core spread: sum of |DD| on the 6 bonds of each column near the inserted core
ring = {}
for (a_, b_), du in dd.items():
    if np.hypot(*P[a_]) < 12:
        ring.setdefault(a_, []).append(abs(du) / B)
top = sorted(ring, key=lambda c_: -sum(ring[c_]))[:3]
for c_ in top:
    print(f"{path}: column (dy, dz) = ({P[c_][0]:+.2f}, {P[c_][1]:+.2f}) A: sum |DD| on its bonds = {sum(ring[c_]):.3f} b ("
          + " ".join(f"{v:.3f}" for v in sorted(ring[c_], reverse=True)) + ")")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
colors = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical slots 1-3 (reference palette)
fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
sel = np.hypot(*(P - cen).T) < 13
for kk in range(3):
    m = sel & (h == kk)
    ax.scatter(P[m, 1], P[m, 0], s=70, facecolor=colors[kk], edgecolor="white", linewidth=1.5, zorder=2,
               label=f"column height {kk}b/3 along [111]")
for (a_, b_), du in dd.items():
    if a_ > b_ or max(np.hypot(*(P[a_] - cen)), np.hypot(*(P[b_] - cen))) > 12.5 or abs(du) < 0.02 * B:
        continue
    v = (P[b_] - P[a_]) if du > 0 else (P[a_] - P[b_])
    Ln = abs(du) / (B / 3)
    mid = (P[a_] + P[b_]) / 2
    st, en = mid - v * Ln / 2, mid + v * Ln / 2
    ax.annotate("", xy=(en[1], en[0]), xytext=(st[1], st[0]), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color="#1a1a19", lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=0))
ax.plot(0, 0, marker="+", color="#1a1a19", ms=10, mew=1.5, zorder=4, label="initial (inserted) core position")
ax.plot(cen[1], cen[0], marker="x", color="#d03b3b", ms=9, mew=1.5, zorder=4, label="Burgers triangle centre (this state)")
ax.scatter(P[top[0], 1], P[top[0], 0], s=260, facecolor="none", edgecolor="#d03b3b", linewidth=1.5, zorder=4,
           label=f"column with largest |DD| sum ({sum(ring[top[0]]):.2f} b)")
ax.set_aspect("equal")
ax.set_xlim(cen[1] - 13, cen[1] + 13); ax.set_ylim(cen[0] - 13, cen[0] + 13)
ax.set_xlabel("z = [-1 -1 2]  glide direction (Å, from inserted core)")
ax.set_ylabel("y = [-1 1 0]  glide-plane normal (Å, from inserted core)")
ax.set_title(f"{title}\ndifferential displacement map, arrow = |Δu| along [111], b/3 = full spacing", fontsize=9)
for sp in ax.spines.values():
    sp.set_color("#c3c2b7")
ax.tick_params(colors="#555", labelsize=8)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(out)
print(f"saved {out}")
