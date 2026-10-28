"""Write a LAMMPS data file with a new box (Lx, Lz, xz), mapping the atoms affinely in x and z; y is left untouched.

Replaces `change_box ... remap` in tools/in.step_chi (Session 11): with the shrink-wrapped y boundary, every remap
operation stretched all y coordinates by ~1e-4 (0.0148 A at the top surface) even with an identical box, which kicked
every step (fmax 2.3e-3 eV/A per operation) and pushed the core over the last small barrier near tau_P.
Map (xlo = zlo = 0, xy = yz = 0): unwrapped r_u = r + ix*(Lx,0,0) + iz*(xz,0,Lz) (old box); fractions lz = z_u/Lz,
lx = (x_u - xz*lz)/Lx; new x_u' = Lx'*lx + xz'*lz, z_u' = Lz'*lz. Written unwrapped with image flags 0; read_data wraps
them into the new box and sets the image flags. The Velocities section is dropped (0 K minimisation).
Usage: python rebox.py <in.data> <out.data> <Lx> <Lz> <xz>     (import: rebox(path_in, path_out, Lx, Lz, xz))
"""
import sys


def rebox(src, dst, Lx_new, Lz_new, xz_new):
    Lx_new, Lz_new, xz_new = float(Lx_new), float(Lz_new), float(xz_new)   # plain floats: repr of numpy floats is 'np.float64(..)'
    L = open(src).read().splitlines()
    i_xlo = next(i for i, l in enumerate(L) if l.rstrip().endswith("xlo xhi"))
    i_zlo = next(i for i, l in enumerate(L) if l.rstrip().endswith("zlo zhi"))
    i_tilt = next(i for i, l in enumerate(L) if l.rstrip().endswith("xy xz yz"))
    xlo, xhi = map(float, L[i_xlo].split()[:2]); zlo, zhi = map(float, L[i_zlo].split()[:2])
    xy, xz, yz = map(float, L[i_tilt].split()[:3])
    assert xlo == 0.0 and zlo == 0.0 and xy == 0.0 and yz == 0.0, "rebox assumes xlo = zlo = 0 and xy = yz = 0"
    Lx, Lz = xhi - xlo, zhi - zlo
    n = int(next(l for l in L if l.strip().endswith("atoms")).split()[0])
    i0 = L.index("Atoms # atomic") + 2
    out = L[:i0]
    out[i_xlo] = f"0 {Lx_new!r} xlo xhi"
    out[i_zlo] = f"0 {Lz_new!r} zlo zhi"
    out[i_tilt] = f"0 {xz_new!r} 0 xy xz yz"
    for l in L[i0:i0 + n]:
        f = l.split()
        x, y, z = map(float, f[2:5])
        ix, iy, iz = (map(int, f[5:8]) if len(f) >= 8 else (0, 0, 0))
        xu, zu = x + ix * Lx + iz * xz, z + iz * Lz
        lz = zu / Lz
        lx = (xu - xz * lz) / Lx
        out.append(f"{f[0]} {f[1]} {Lx_new * lx + xz_new * lz!r} {y!r} {Lz_new * lz!r} 0 {iy} 0")
    open(dst, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    rebox(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
