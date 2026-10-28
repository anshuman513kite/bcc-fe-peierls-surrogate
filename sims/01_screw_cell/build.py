"""Build a slab with one straight 1/2[111] screw dislocation in bcc Fe (Mendelev 2003), for LAMMPS.

Frame (relabelled from Ansh's spec so LAMMPS can tilt the box; same physics):
  x = [1 1 1]   dislocation line, periodic, 3 lattice repeats (a*sqrt3 each = 2b)
  y = [-1 1 0]  glide-plane normal, free surfaces
  z = [-1 -1 2] glide direction, periodic
Boundary-mismatch method: periodic array of dislocations (PAD, Osetsky & Bacon 2003).
  u_x = (b/2pi) * arg( sin( pi*((z-z0) + i(y-y0)) / Lz ) )
  This is the isotropic screw field b/(2pi)*atan2(y-y0, z-z0) summed over its periodic images along z.
  Across the z boundary it jumps by b/2 (mod b) in both half-crystals; a box tilt xz = b/2 absorbs that exactly.

Usage: [SCREW_NZ=42] python build.py <core: up|down> <out.data>  -> also writes perfect positions to <out>.ref.npy
"""
import os
import sys
import numpy as np

A = 2.8553                    # lattice constant, Angstrom (checks/01_lattice, 4 decimals as specified by Ansh)
B = A * np.sqrt(3) / 2        # Burgers vector magnitude
NX, NY = 3, 37                # lattice repeats along x, y
NZ = int(os.environ.get("SCREW_NZ", 21))   # repeats along z (glide); 21 = 146.87 A, 42 = 2x cell

ex = np.array([1, 1, 1]) / np.sqrt(3)
ey = np.array([-1, 1, 0]) / np.sqrt(2)
ez = np.array([-1, -1, 2]) / np.sqrt(6)
LX, LY, LZ = NX * A * np.sqrt(3), NY * A * np.sqrt(2), NZ * A * np.sqrt(6)


def perfect_lattice():
    """All bcc sites inside [0,LX) x [0,LY) x [0,LZ) in the rotated frame.
    Scanned one cubic i-plane at a time: the full grid at once needs > 10 GB for the 2x cell (NZ = 42)
    and ran the laptop out of memory. Same sites in the same order as the original all-at-once version."""
    n = int(np.ceil(max(LX, LY, LZ) * 2 / A)) + 2
    g = np.arange(-n, n + 1)
    j, k = np.meshgrid(g, g, indexing="ij")
    jk = np.stack([j.ravel(), k.ravel()], 1).astype(float)
    tol = 1e-6
    out = []
    for shift in (0.0, 0.5):                     # cube corners first, then body centres (original order)
        for i in g:
            cub = (np.column_stack([np.full(len(jk), float(i)), jk]) + shift) * A
            rot = np.stack([cub @ ex, cub @ ey, cub @ ez], 1)
            keep = ((rot[:, 0] >= -tol) & (rot[:, 0] < LX - tol) & (rot[:, 1] >= -tol) & (rot[:, 1] < LY - tol)
                    & (rot[:, 2] >= -tol) & (rot[:, 2] < LZ - tol))
            out.append(rot[keep])
    return np.vstack(out)


def core_position(pos, kind):
    """Centroid of a triangle of three nearest [111] columns near the box centre (projected on y-z).
    'up'/'down' = the two triangle orientations; one is the easy core, the other the hard core for a given b sign."""
    cols = np.unique(np.round(pos[:, 1:], 4), axis=0)
    c = np.array([LY / 2, LZ / 2])
    d = np.linalg.norm(cols - c, axis=1)
    p0 = cols[np.argmin(d)]
    dnn = A * np.sqrt(2 / 3)                    # nearest column spacing in the projection
    nb = cols[np.abs(np.linalg.norm(cols - p0, axis=1) - dnn) < 1e-3]
    # pick the two neighbours that form a triangle pointing +y ('up') or -y ('down') with p0
    best = None
    for a_ in range(len(nb)):
        for b_ in range(a_ + 1, len(nb)):
            if abs(np.linalg.norm(nb[a_] - nb[b_]) - dnn) > 1e-3:
                continue
            cen = (p0 + nb[a_] + nb[b_]) / 3
            ys = sorted([p0[0], nb[a_][0], nb[b_][0]])
            pointing_up = abs(ys[1] - ys[0]) < 1e-3          # two at the bottom, apex on top
            if (kind == "up") == pointing_up:
                if best is None or np.linalg.norm(cen - c) < np.linalg.norm(best - c):
                    best = cen
    return best                                  # (y0, z0)


def screw_ux(y, z, y0, z0):
    w = np.pi * ((z - z0) + 1j * (y - y0)) / LZ
    return B / (2 * np.pi) * np.angle(np.sin(w))


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    for n1, v1, n2, v2 in [("x", ex, "y", ey), ("x", ex, "z", ez), ("y", ey, "z", ez)]:
        print(f"dot({n1},{n2}) = {v1 @ v2:+.1e}")
    print(f"right-handed: x cross y = z ? {np.allclose(np.cross(ex, ey), ez)}")
    pos = perfect_lattice()
    print(f"a = {A} A, b = {B:.4f} A")
    print(f"box: Lx(line) = {LX:.4f} A, Ly(normal) = {LY:.4f} A, Lz(glide) = {LZ:.4f} A, atoms = {len(pos)}")
    y0, z0 = core_position(pos, kind)
    print(f"core ({kind} triangle): y0 = {y0:.4f} A, z0 = {z0:.4f} A")
    # PAD periodicity check: u(z+Lz) - u(z) must equal b/2 modulo b for every height y
    yy = np.linspace(0, LY, 201)
    jump = screw_ux(yy, z0 + LZ / 2 - 1e-9, y0, z0) - screw_ux(yy, z0 - LZ / 2 + 1e-9, y0, z0)
    print(f"PAD check: max |(u(z+Lz)-u(z) - b/2) mod b| = {np.abs((jump - B / 2 + B / 2) % B - B / 2).max():.2e} A")
    ux = screw_ux(pos[:, 1], pos[:, 2], y0, z0)
    new = pos.copy()
    new[:, 0] += ux
    xz = B / 2
    new[:, 0] -= np.floor((new[:, 0] - xz * new[:, 2] / LZ) / LX) * LX      # wrap into the tilted cell
    np.save(out + ".ref.npy", pos)
    np.save(out + ".core.npy", np.array([y0, z0]))
    with open(out, "w") as f:
        f.write(f"Fe bcc 1/2[111] screw, PAD field, core {kind} triangle\n\n{len(new)} atoms\n1 atom types\n\n")
        f.write(f"0 {LX:.10f} xlo xhi\n{-5.0:.10f} {LY + 5.0:.10f} ylo yhi\n0 {LZ:.10f} zlo zhi\n")
        f.write(f"0.0 {xz:.10f} 0.0 xy xz yz\n\nMasses\n\n1 55.845\n\nAtoms # atomic\n\n")
        for i, p in enumerate(new, 1):
            f.write(f"{i} 1 {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
