"""Peierls-stress ramp for one screw dislocation (chi = 0) by surface forces, driving LAMMPS step by step.

Usage (from the simulation folder):
  python peierls.py <relaxed.data> <y0> <name> [--coarse 0:1.6:0.05] [--fine 0.01] [--maxit 10000] [--np NP (default: $NP from launch_safe.sh, else cores - 1)]
Writes: steps/ (data per step), log.step_*, ../../data/<name>.csv, ../../logs/<name>.log
Each step starts from the previous relaxed configuration. The core moves on, step to step, once the stress
exceeds the Peierls stress; a step is stopped early (fix halt) once the core has run more than HALT_JUMPS jumps.
"""
import argparse, csv, os, re, subprocess, sys, time
import numpy as np

A0 = 2.8553
JUMP = A0 * np.sqrt(6) / 3          # one Peierls jump along [-1-12] = 2.331 A
GPA_A2_TO_EV_A = 1e9 * 1e-20 / 1.602176634e-19 * 1e-10   # 1 GPa*A^2 in eV/A = 0.0062415
HALT_JUMPS = 3

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("y0", type=float); ap.add_argument("name")
ap.add_argument("--coarse", default="0:1.6:0.05"); ap.add_argument("--fine", type=float, default=0.01)
ap.add_argument("--maxit", type=int, default=10000); ap.add_argument("--np", type=int, default=int(os.environ.get("NP", os.cpu_count() - 1)))
ap.add_argument("--only", type=float, default=None, help="test: run a single step at this stress, then stop")
args = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs("steps", exist_ok=True); os.makedirs(f"{root}/logs", exist_ok=True); os.makedirs(f"{root}/data", exist_ok=True)
plog = open(f"{root}/logs/{args.name}.log", "a")


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True); plog.write(line + "\n"); plog.flush()


# ---- cell geometry and surface slabs (2 atomic (-110) planes at each free surface) ----
L = open(args.data).read().splitlines()
hdr = {k: [float(v) for v in l.split()[:2]] for l in L[:30] for k in ("xlo", "zlo") if l.rstrip().endswith(f"{k} {k[0]}hi")}
LX, LZ = hdr["xlo"][1] - hdr["xlo"][0], hdr["zlo"][1] - hdr["zlo"][0]
n_atoms = int(next(l for l in L if l.strip().endswith("atoms")).split()[0])
i0 = L.index("Atoms # atomic") + 2
atoms = np.array([l.split()[:5] for l in L[i0:i0 + n_atoms]], float)
ids, y = atoms[:, 0].astype(int), atoms[:, 3]
d110 = A0 * np.sqrt(2) / 2
top, bot = ids[y > y.max() - 1.5 * d110], ids[y < y.min() + 1.5 * d110]
def planes(v):                                  # mean y of the two atomic planes in a slab
    cut = (v.min() + v.max()) / 2
    return f"{v[v < cut].mean():.3f}, {v[v >= cut].mean():.3f} A ({np.count_nonzero(v < cut)} + {np.count_nonzero(v >= cut)} atoms)"
ytop, ybot = planes(y[y > y.max() - 1.5 * d110]), planes(y[y < y.min() + 1.5 * d110])
assert len(top) == len(bot)
with open("groups.lmp", "w") as f:
    f.write("group top id " + " ".join(map(str, top)) + "\ngroup bot id " + " ".join(map(str, bot)) + "\n")
AREA = LX * LZ
say(f"cell: Lx = {LX:.4f} A, Lz = {LZ:.4f} A, atoms = {n_atoms}, glide plane y0 = {args.y0:.4f} A")
say(f"slabs: top {len(top)} atoms in planes y = {ytop} ; bottom {len(bot)} atoms in planes y = {ybot}")
say(f"slab = 2 (-110) planes, spacing d110 = {d110:.4f} A -> nominal thickness 2*d110 = {2 * d110:.3f} A")
say(f"hand check: A = Lx*Lz = {AREA:.2f} A^2 ; 1 GPa*A^2 = {GPA_A2_TO_EV_A:.6f} eV/A ; "
    f"total force per surface at 1 GPa = {AREA * GPA_A2_TO_EV_A:.3f} eV/A ; per atom = {AREA * GPA_A2_TO_EV_A / len(top):.6f} eV/A")


def run_step(sigma, datain, kind):
    tag = f"{kind}_{sigma:.3f}"
    fx = sigma * AREA * GPA_A2_TO_EV_A / len(top)
    dataout = f"steps/{tag}.data"
    cmd = ["mpirun", "-np", str(args.np), "lmp", "-in", "in.step", "-screen", "none", "-log", f"steps/log.{tag}",
           "-var", "datain", datain, "-var", "dataout", dataout, "-var", "fx", f"{fx:.10f}", "-var", "y0", f"{args.y0}",
           "-var", "LZ", f"{LZ:.10f}", "-var", "maxit", str(args.maxit), "-var", "groups", "groups.lmp",
           "-var", "halt_dist", f"{HALT_JUMPS * JUMP:.4f}"]
    t = time.time()
    rc = subprocess.run(cmd).returncode
    log = open(f"steps/log.{tag}").read()
    if rc != 0 or "END " not in log:
        say(f"LAMMPS failed at sigma = {sigma:.3f} GPa (rc {rc})"); sys.exit(1)
    end = dict(kv.split("=") for kv in re.search(r"^END (.*)$", log, re.M).group(1).split())
    start = dict(kv.split("=") for kv in re.search(r"^START (.*)$", log, re.M).group(1).split())
    m = re.search(r"Stopping criterion = (.*)", log)
    stop = m.group(1).strip() if m else "halted (core ran > %d jumps)" % HALT_JUMPS
    if "Fix halt condition" in log:              # LAMMPS reports a fix-halt stop as "walltime limit reached"
        stop = "halted (core ran > %d jumps)" % HALT_JUMPS
    it = re.search(r"Iterations, force evaluations = (\d+) (\d+)", log)
    conv = stop in ("energy tolerance", "force tolerance")
    return dict(kind=kind, sigma_GPa=sigma, fx_eV_per_A=fx, E_total_eV=float(end["pe"]), E_pair_eV=float(end["epair"]),
                core_z_A=float(end["cz"]) % LZ, core_y_A=float(end["cy"]), dz_step_A=float(end["dcz"]),
                fmax=float(end["fmax"]), iterations=int(it.group(1)) if it else -1, stop=stop,
                converged="yes" if conv else "no", seconds=round(time.time() - t, 1), data=dataout)


rows, t0 = [], time.time()
lo, hi, st = map(float, args.coarse.split(":"))
coarse = [round(s, 4) for s in np.arange(lo, hi + st / 2, st)]
if args.only is not None:
    coarse = [args.only]
ztot = 0.0                       # cumulative core displacement along z (unwrapped), from the start of the ramp


def record(r):
    global ztot
    ztot += r["dz_step_A"]
    r["core_z_cum_A"] = round(ztot, 4)
    rows.append(r)
    el = time.time() - t0
    say(f"step {len(rows)} [{r['kind']}] sigma = {r['sigma_GPa']:.3f} GPa | dz_step = {r['dz_step_A']:+.3f} A | "
        f"core z = {r['core_z_A']:.3f} A (cumulative {ztot:+.3f}) | core y = {r['core_y_A']:.3f} | "
        f"converged: {r['converged']} ({r['stop']}, {r['iterations']} it, fmax {r['fmax']:.1e}) | {r['seconds']} s | elapsed {el:.0f} s")


def moving(r):
    return abs(r["dz_step_A"]) >= JUMP


say(f"ramp: coarse {coarse[0]}..{coarse[-1]} GPa step {st}, then {args.fine} GPa refinement; jump = {JUMP:.4f} A")
prev, sig_move, last_stable_data, last_stable_sigma, last_stable_z = args.data, None, args.data, None, 0.0
for k, s in enumerate(coarse):
    r = run_step(s, prev, "coarse"); record(r)
    say(f"progress: coarse step {k + 1} / {len(coarse)} (max) | elapsed {time.time() - t0:.0f} s | "
        f"est. left <= {(time.time() - t0) / (k + 1) * (len(coarse) - k - 1):.0f} s if no step runs away")
    if moving(r) and r["converged"] == "no":       # moved >= 1 jump and was still going when stopped
        sig_move = s; break
    if not moving(r):
        last_stable_data, last_stable_sigma, last_stable_z = r["data"], s, ztot
    prev = r["data"]

if sig_move is not None and last_stable_sigma is not None:
    say(f"coarse bracket: stable {last_stable_sigma:.3f} GPa, moving {sig_move:.3f} GPa -> refining")
    fine = [round(last_stable_sigma + args.fine * j, 4) for j in range(1, int(round((sig_move - last_stable_sigma) / args.fine)))]
    ztot = last_stable_z                                           # refinement continues from the last stable config
    prev, fine_move, fine_stable = last_stable_data, sig_move, last_stable_sigma
    for s in fine:
        r = run_step(s, prev, "fine"); record(r)
        if moving(r) and r["converged"] == "no":
            fine_move = s; break
        if not moving(r):
            fine_stable = s
        prev = r["data"]
    say(f"RESULT {args.name}: Peierls stress between {fine_stable:.3f} GPa (last stable) and {fine_move:.3f} GPa (first moving)")
    say(f"RESULT {args.name}: core moved {'+z' if rows[-1]['dz_step_A'] > 0 else '-z'} under +sigma_xy")
elif args.only is None:
    say(f"RESULT {args.name}: no Peierls bracket found (sig_move = {sig_move}, last stable = {last_stable_sigma})")

with open(f"{root}/data/{args.name}.csv", "w", newline="") as f:
    keys = ["kind", "sigma_GPa", "E_total_eV", "E_pair_eV", "core_z_A", "core_z_cum_A", "core_y_A", "dz_step_A",
            "converged", "stop", "iterations", "fmax", "fx_eV_per_A", "seconds"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
say(f"done: {len(rows)} steps, total {time.time() - t0:.0f} s; csv data/{args.name}.csv")
