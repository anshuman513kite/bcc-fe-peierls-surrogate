"""Numerically compare two single-element eam/fs (setfl) files: grid line and all tabulated values.

Usage: python compare_eamfs.py ours.eam.fs other.eam.fs
Skips the 3 free-text comment lines; compares the element line, the grid line and every number after it.
"""
import sys
import numpy as np

def load(path):
    lines = open(path).read().splitlines()
    elem = lines[3].split()
    grid = [float(v) for v in lines[4].split()]
    nrho, nr = int(grid[0]), int(grid[2])
    rest = " ".join(lines[5:]).split()
    elem_line = rest[:4]          # Z, mass, a0, lattice
    vals = np.array([float(v.replace("D", "E")) for v in rest[4:]])
    return elem, grid, elem_line, vals, nrho, nr

a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"element line : {a[0]} vs {b[0]} | {a[2]} vs {b[2]}")
print(f"grid         : {a[1]}\n               {b[1]}  -> {'same' if a[1] == b[1] else 'DIFFERENT'}")
print(f"value counts : {a[3].size} vs {b[3].size} (expected Nrho + 2*Nr = {a[4] + 2 * a[5]})")
if a[1] == b[1] and a[3].size == b[3].size:
    d = np.abs(a[3] - b[3])
    scale = np.maximum(np.abs(a[3]), np.abs(b[3]))
    rel = np.where(scale > 0, d / np.where(scale > 0, scale, 1), 0)
    for name, sl in [("F(rho)", slice(0, a[4])), ("rho(r)", slice(a[4], a[4] + a[5])), ("r*phi(r)", slice(a[4] + a[5], None))]:
        print(f"  {name:8s}: max |diff| = {d[sl].max():.3e}, max rel diff = {rel[sl].max():.3e}, identical values: {np.count_nonzero(d[sl] == 0)}/{d[sl].size}")
    ok = a[2] == b[2] and rel.max() <= 1e-14
    print("RESULT:", "MATCH (same grid, same values to file precision)" if ok else "NO MATCH")
else:
    print("RESULT: NO MATCH (grid or table size differs)")
