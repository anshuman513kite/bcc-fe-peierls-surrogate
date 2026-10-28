"""Analyse a relaxed screw cell: far-field energy per atom, far-field shear strain, core symmetry, DD map.

Usage: python analyze.py <tag> [ddmap.png]
Reads screw_<tag>.data.ref.npy / .core.npy (perfect positions, core), relaxed_<tag>.dump, inserted_<tag>.dump.
Frame: x = [111] line, y = [-110] glide-plane normal, z = [-1-12] glide direction.
"""
import sys
import numpy as np
from build import A, B, LX, LZ, screw_ux

tag = sys.argv[1]
ref = np.load(f"screw_{tag}.data.ref.npy")
y0, z0 = np.load(f"screw_{tag}.data.core.npy")
XZ = B / 2


def load_dump(path):
    lines = open(path).read().splitlines()
    i = lines.index(next(l for l in lines if l.startswith("ITEM: ATOMS")))
    d = np.loadtxt(lines[i + 1:])
    return d[d[:, 0].argsort()]


def line_disp(d):
    """Displacement along the line (x) relative to the perfect crystal, mod b, in [-b/2, b/2)."""
    x, z = d[:, 1].copy(), d[:, 3].copy()
    n = np.round((z - ref[:, 2]) / LZ)           # atoms wrapped across z pick up the tilt xz
    x -= n * XZ
    return (x - ref[:, 0] + B / 2) % B - B / 2


rel, ins = load_dump(f"relaxed_{tag}.dump"), load_dump(f"inserted_{tag}.dump")
assert len(rel) == len(ref)
pe = rel[:, 4]
ys = rel[:, 2]
dz = (ref[:, 2] - z0 + LZ / 2) % LZ - LZ / 2      # minimum-image distance to core along z
r = np.hypot(ref[:, 1] - y0, dz)

# (3) energy per atom far from core and surfaces
far = (r > 40) & (ys > ys.min() + 15) & (ys < ys.max() - 15)
print(f"RESULT {tag} total E = {pe.sum():.6f} eV, N = {len(pe)}")
print(f"RESULT {tag} far-field atoms (r>40 A from core, >15 A from both surfaces): n = {far.sum()}, "
      f"pe mean = {pe[far].mean():.5f}, min = {pe[far].min():.5f}, max = {pe[far].max():.5f} eV/atom")
for lo, hi in [(40, 50), (50, 60), (60, 70)]:
    m = (r > lo) & (r <= hi) & (ys > ys.min() + 15) & (ys < ys.max() - 15)
    print(f"        r in ({lo},{hi}] A: n = {m.sum()}, pe mean = {pe[m].mean():.5f} eV/atom")

# far-field shear strain gamma = d(u_x)/dz in the upper and lower half-crystals (PAD long-range field)
ux = line_disp(rel)
for name, band in [("upper", (ref[:, 1] > y0 + 30) & (ys < ys.max() - 15)),
                   ("lower", (ref[:, 1] < y0 - 30) & (ys > ys.min() + 15))]:
    band &= np.abs(dz) < LZ / 2 - 10              # stay away from the z cut of the PAD field
    pred = screw_ux(ref[band, 1], ref[band, 2], y0, z0)
    res = (ux[band] - pred + B / 2) % B - B / 2
    u = pred + res
    slope = np.polyfit(ref[band, 2], u, 1)[0]
    print(f"RESULT {tag} {name} half far field: gamma_xz = du_x/dz = {slope:+.5f} "
          f"(isotropic PAD formula -+b/(2Lz) = {-B / (2 * LZ) if name == 'upper' else B / (2 * LZ):+.5f}), n = {band.sum()}")

# columns: one atom per [111] column (perfect x in [0, b))
col = (ref[:, 0] >= -1e-6) & (ref[:, 0] < B - 1e-6)
P = ref[col][:, 1:] - [y0, z0]                    # projected positions relative to core (y, z)
h = np.round(ref[col][:, 0] / (B / 3)).astype(int) % 3
near = np.hypot(*P.T) < 14
idx = np.where(near)[0]
dnn = A * np.sqrt(2 / 3)


def ddmap(ux_all):
    u = ux_all[col]
    pairs = []
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            i, j = idx[a_], idx[b_]
            if np.linalg.norm(P[i] - P[j]) < dnn + 0.1:
                du = (u[j] - u[i] + B / 2) % B - B / 2
                pairs.append((i, j, du))
    return pairs


pairs = ddmap(ux)
print(f"columns in projection: {col.sum()} (expected N/6 = {len(ref) // 6})")


def key(p):
    return tuple(np.round(p, 2))


lookup = {}
for i, j, du in pairs:
    lookup[(key(P[i]), key(P[j]))] = du
    lookup[(key(P[j]), key(P[i]))] = -du

# core-centre triangle: the three columns closest to the core
c3 = np.argsort(np.hypot(*P.T))[:3]
tri = [abs(lookup[(key(P[i]), key(P[j]))]) / B for i, j in [(c3[0], c3[1]), (c3[1], c3[2]), (c3[2], c3[0])]]
print(f"RESULT {tag} central-triangle |DD| / b = {tri[0]:.4f}, {tri[1]:.4f}, {tri[2]:.4f}  (1/3 = 0.3333)")

# symmetry checks on |DD| within 10 A of the core: 3-fold rotation and the [-110] diad (mirror z -> -z in projection)
R = np.array([[np.cos(2 * np.pi / 3), -np.sin(2 * np.pi / 3)], [np.sin(2 * np.pi / 3), np.cos(2 * np.pi / 3)]])
for name, op in [("3-fold rotation", lambda p: R @ p), ("diad [-110] (z -> -z)", lambda p: p * [1, -1])]:
    dev, n = 0.0, 0
    for i, j, du in pairs:
        if max(np.hypot(*P[i]), np.hypot(*P[j])) > 10:
            continue
        k = (key(op(P[i])), key(op(P[j])))
        if k in lookup:
            dev = max(dev, abs(abs(du) - abs(lookup[k]))); n += 1
    print(f"RESULT {tag} {name}: max | |DD| - |DD(image)| | = {dev / B:.4f} b over {n} pairs")

if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]    # categorical slots 1-3 (reference palette)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
    sel = np.hypot(*P.T) < 13
    for k in range(3):
        m = sel & (h == k)
        ax.scatter(P[m, 1], P[m, 0], s=70, facecolor=colors[k], edgecolor="white", linewidth=1.5, zorder=2,
                   label=f"column height {k}b/3 along [111]")
    for i, j, du in pairs:
        if max(np.hypot(*P[i]), np.hypot(*P[j])) > 12.5 or abs(du) < 0.02 * B:
            continue
        v = (P[j] - P[i]) if du > 0 else (P[i] - P[j])
        L = abs(du) / (B / 3)                      # arrow length: |DD| = b/3 spans the full column spacing
        mid = (P[i] + P[j]) / 2
        s, e = mid - v * L / 2, mid + v * L / 2
        ax.annotate("", xy=(e[1], e[0]), xytext=(s[1], s[0]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color="#1a1a19", lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.plot(0, 0, marker="+", color="#1a1a19", ms=10, mew=1.5, zorder=4, label="initial core position")
    ax.set_aspect("equal")
    ax.set_xlabel("z = [-1 -1 2]  glide direction (Å, from core)")
    ax.set_ylabel("y = [-1 1 0]  glide-plane normal (Å, from core)")
    ax.set_title("Relaxed 1/2[111] screw core, Fe (Mendelev 2003 #2)\n"
                 "differential displacement map, arrow = |Δu| along [111], b/3 = full spacing", fontsize=9)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(sys.argv[2])
    print(f"saved {sys.argv[2]}")
