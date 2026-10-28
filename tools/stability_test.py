"""Stability test of a relaxed screw-core state under load (Gate B', Session 11).

Usage (from a folder holding screw_down.data.ref.npy / .core.npy of the cell; launch with tools/launch_safe.sh):
  python stability_test.py <state.data> <tau> <chi> <y0> <tag> [--seed 1] [--np 7] [--nz 21] [--model strain_model.json]
       [--jiggle 0.001] [--etol 0] [--ftol 1e-6] [--dmax 0.01] [--maxit 200000] [--maxeval 2000000]
The state is displaced randomly (uniform in [-jiggle, jiggle] A per Cartesian component), then re-minimised at the same
loading (tools/in.step_chi; for chi = 0 that is the Session 5 loading: surface forces only, box unchanged).
The core site before and after comes from the DD core finder (tools/ddmap_data.py: Burgers triangle centroid and the column
with the largest |DD| sum). "returned" (D5, Ansh, Session 14; label-free, tools/moved_rule.same_state) = still compact at
the same site (top column < 1.4 b), or still column-centred on the same column for a split state = a true resting state.
The old label-based check (same Burgers triangle) is printed too, as returned_triangle.
Prints one line: STABILITY <tag> tau=.. chi=.. np=.. seed=.. stop=.. it=.. fmax=.. fnorm=.. before=(dy,dz) after=(dy,dz) returned=yes/no
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moved_rule import dd_state, same_state
from rebox import rebox
A0 = 2.8553
GPA_A2_TO_EV_A = 1e9 * 1e-20 / 1.602176634e-19 * 1e-10
LIN = np.array([0.0, 0.0027625, 0.0164349])
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("tau", type=float); ap.add_argument("chi", type=float); ap.add_argument("y0", type=float)
ap.add_argument("tag")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--np", type=int, default=int(os.environ.get("NP", 7)))
ap.add_argument("--nz", type=int, default=21); ap.add_argument("--model", default=None)
ap.add_argument("--ref", default=None, help="unstrained reference data file for the box (default: the state itself, chi = 0)")
ap.add_argument("--jiggle", default="0.001"); ap.add_argument("--etol", default="0"); ap.add_argument("--ftol", default="1e-6")
ap.add_argument("--dmax", default="0.01"); ap.add_argument("--maxit", type=int, default=200000); ap.add_argument("--maxeval", type=int, default=2000000)
a = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs("stab", exist_ok=True)


def header(path):
    L = open(path).read().splitlines()
    g = lambda k: [float(v) for v in next(l for l in L[:30] if l.rstrip().endswith(k)).split()[:2]]
    return L, g("xlo xhi"), g("zlo zhi"), float(next(l for l in L[:30] if l.rstrip().endswith("xy xz yz")).split()[1])


L, (xlo, xhi), (zlo, zhi), xz_state = header(a.data)
_, (rxlo, rxhi), (rzlo, rzhi), XZ0 = header(a.ref or a.data)
LX0, LZ0 = rxhi - rxlo, rzhi - rzlo
n = int(next(l for l in L if l.strip().endswith("atoms")).split()[0]); i0 = L.index("Atoms # atomic") + 2
at = np.array([l.split()[:5] for l in L[i0:i0 + n]], float)
ids, y = at[:, 0].astype(int), at[:, 3]
d110 = A0 * np.sqrt(2) / 2
top, bot = ids[y > y.max() - 1.5 * d110], ids[y < y.min() + 1.5 * d110]
with open(f"stab/groups_{a.tag}.lmp", "w") as f:
    f.write("group top id " + " ".join(map(str, top)) + "\ngroup bot id " + " ".join(map(str, bot)) + "\n")
chi = np.radians(a.chi)
if a.model:                       # Session 13: also at chi = 0 (the sweep protocol applies the chi = 0 correction; Gate B' passed no --model)
    m = json.load(open(a.model))["correction"][str(int(round(a.chi)))]
    Q = np.array([m["q2"], m["q3"], m["q4"]])
else:
    Q = np.zeros((3, 4))
c = Q[0] * a.tau ** 2 + Q[1] * a.tau ** 3 + Q[2] * a.tau ** 4
e = LIN * a.tau * np.sin(chi) + c[:3]; sxy = a.tau * np.cos(chi) + c[3]
Lx, Lz = LX0 * (1 + e[0]), LZ0 * (1 + e[1]); xz = XZ0 + e[2] * Lz
fx = sxy * Lx * Lz * GPA_A2_TO_EV_A / len(top)


def dd(path):
    out = subprocess.run([sys.executable, f"{root}/tools/ddmap_data.py", path, f"stab/dd_{os.path.basename(path)}.png"],
                         capture_output=True, text=True, env=dict(os.environ, SCREW_NZ=str(a.nz)), check=True).stdout
    m = re.search(r"dy = (\S+) A, dz = (\S+) A", out); col = re.search(r"column \(dy, dz\) = \((\S+), (\S+)\) A: sum \|DD\| on its bonds = (\S+) b", out)
    return (float(m.group(1)), float(m.group(2))), (float(col.group(1).rstrip(",")), float(col.group(2)), float(col.group(3)))


before, colb = dd(a.data)
datain = a.data
if max(abs(xhi - xlo - Lx), abs(zhi - zlo - Lz), abs(xz_state - xz)) > 1e-12:     # box differs from the loading: rebox
    datain = f"stab/{a.tag}.in.data"; rebox(a.data, datain, Lx, Lz, xz)
y0c, z0c = np.load("screw_down.data.core.npy")
out = f"stab/{a.tag}.data"
cmd = ["mpirun", "-np", str(a.np), "lmp", "-in", f"{root}/tools/in.step_chi", "-screen", "none", "-log", f"stab/log.{a.tag}",
       "-var", "pot", f"{root}/potentials/Fe_mm.eam.fs", "-var", "datain", datain, "-var", "dataout", out, "-var", "peout", f"stab/{a.tag}.pe",
       "-var", "fx", f"{fx:.10f}", "-var", "y0", f"{a.y0}", "-var", "Lx", f"{Lx:.10f}", "-var", "Lz", f"{Lz:.10f}", "-var", "xz", f"{xz:.10f}",
       "-var", "ys", f"{y0c + before[0]:.6f}", "-var", "zs", f"{(z0c + before[1]) * (1 + e[1]):.6f}", "-var", "maxit", str(a.maxit),
       "-var", "groups", f"stab/groups_{a.tag}.lmp", "-var", "halt_dist", f"{3 * A0 * np.sqrt(6) / 3:.4f}",
       "-var", "etol", a.etol, "-var", "ftol", a.ftol, "-var", "dmax", a.dmax, "-var", "maxeval", str(a.maxeval),
       "-var", "jiggle", a.jiggle, "-var", "seed", str(a.seed)]
rc = subprocess.run(cmd).returncode
log = open(f"stab/log.{a.tag}").read()
if rc != 0 or "END " not in log:
    print(f"STABILITY {a.tag} LAMMPS failed rc={rc}"); sys.exit(1)
end = dict(kv.split("=") for kv in re.search(r"^END (.*)$", log, re.M).group(1).split())
ms = re.search(r"Stopping criterion = (.*)", log); stop = ms.group(1).strip() if ms else "halted"
if "Fix halt condition" in log:
    stop = "halted (core left)"
it = re.search(r"Iterations, force evaluations = (\d+)", log)
after, cola = dd(out)
ret_tri = "yes" if np.hypot(after[0] - before[0], after[1] - before[1]) < 0.01 else "no"
sb, sa = dd_state(a.data, ".", a.nz, "stab/_dd.png"), dd_state(out, ".", a.nz, "stab/_dd.png")
ret = "yes" if same_state(sb, sa) else "no"
print(f"STABILITY {a.tag} tau={a.tau:.3f} chi={a.chi:+.0f} np={a.np} seed={a.seed} stop='{stop}' it={it.group(1) if it else -1} "
      f"fmax={float(end['fmax']):.2e} fnorm={float(end['fnorm']):.2e} before=({before[0]:+.3f},{before[1]:+.3f}) "
      f"after=({after[0]:+.3f},{after[1]:+.3f}) topcol_before=({sb['col'][0]:+.2f},{sb['col'][1]:+.2f},{sb['topcol']:.3f}b) topcol_after=({cola[0]:+.2f},{cola[1]:+.2f},{cola[2]:.3f}b) "
      f"returned_triangle={ret_tri} returned={ret}", flush=True)
