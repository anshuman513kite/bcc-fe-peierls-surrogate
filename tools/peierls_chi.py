"""Peierls-stress ramp at MRSS angle chi (screw dislocation, PROVISIONAL option-2 loading, Session 10), driving LAMMPS step by step.

Usage (from the run folder; launch with tools/launch_safe.sh):
  python peierls_chi.py <relaxed.data> <y0> <z0> <name> --chi DEG [--model strain_model.json] [--nocorr]
                        [--coarse 0:2.4:0.05] [--fine 0.01] [--maxit 10000] [--np NP] [--nz 21]
<relaxed.data> must be the UNSTRAINED relaxed cell (its box is the reference for the box strains); (y0, z0) = inserted core.
Loading at applied tau (GPa): sigma_xy = tau*cos(chi) by surface forces, sigma_xz = tau*sin(chi) by box strains
  eps_xx, eps_zz, gamma_xz = LIN*tau*sin(chi) + q2 tau^2 + q3 tau^3 + q4 tau^4 (checks/07_chi_calibration/strain_model.json),
  surface-force stress = tau*cos(chi) + (same polynomial, column 3). --nocorr: linear strains only and surface stress tau*cos(chi)
  (for chi = 0 this is exactly the Session 5 loading: no box strain).
Peierls stress (Ansh's definition): first tau at which no stable state remains in the starting valley. Detected (default
  --moved dd) from the DD map of each relaxed step (tools/ddmap_data.py) with rule D5 (Ansh, Session 14, tools/moved_rule.py):
  moved = the compact easy core at the start site is lost: a glide jump (>= one jump to a compact 'down' site) or an
  easy -> split transition (top column >= 1.4 b); column `event`. --moved dd_d1: rule D1 (Session 13; label-dependent),
  --moved dd_s12: the Session 11-12 rule (any site change > 0.01 A); both kept only to reproduce old runs. Before the fix (--moved tracker): tools/track_core.py shift >= one jump (2.331 A), which missed a
  one-site jump at chi = -30/-24 because the energy centroid moved only ~1.98 A. The tracker still sets the 6 A window. Coarse steps until the first move, then 10 MPa steps from the last unmoved state.
Each step starts from the previous relaxed configuration. Writes steps/, ../../data/<name>.csv, ../../logs/<name>.log,
and DD maps (tools/ddmap_data.py) of the last unmoved and first moved states in ../../figures/ddmap_<name>_*.png.
"""
import argparse, csv, json, os, re, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebox import rebox
from moved_rule import classify, classify_d1, is_moved, dd_state, EVENT_NAME

A0 = 2.8553
JUMP = A0 * np.sqrt(6) / 3
GPA_A2_TO_EV_A = 1e9 * 1e-20 / 1.602176634e-19 * 1e-10
HALT_JUMPS = 3
LIN = np.array([0.0, 0.0027625, 0.0164349])          # (eps_xx, eps_zz, gamma_xz) per GPa of pure sigma_xz, slab (Session 9)

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("y0", type=float); ap.add_argument("z0", type=float); ap.add_argument("name")
ap.add_argument("--chi", type=float, required=True)
ap.add_argument("--model", default=None); ap.add_argument("--nocorr", action="store_true")
ap.add_argument("--coarse", default="0:2.4:0.05"); ap.add_argument("--fine", type=float, default=0.01)
ap.add_argument("--maxit", type=int, default=10000)
ap.add_argument("--np", type=int, default=int(os.environ.get("NP", os.cpu_count() - 1)))
ap.add_argument("--nz", type=int, default=21)
ap.add_argument("--etol", default="1e-12"); ap.add_argument("--ftol", default="1e-12"); ap.add_argument("--dmax", default="0.1")
ap.add_argument("--maxeval", type=int, default=100000)
ap.add_argument("--ddmaps", default="yes", help="no: skip DD maps at the bracket (stability tests do their own)")
ap.add_argument("--ref", default=None, help="unstrained reference cell for the box strains and surface slabs (default: data); "
                "set it when <data> is an already loaded state (continuing a ramp)")
ap.add_argument("--dd0", default=None, help="start site 'dy,dz' (A, relative to the inserted core) for the moved test (default: the "
                "site of the first step); '0,0' when continuing a ramp whose start was the inserted easy core")
ap.add_argument("--start_core", default=None, help="tracker start 'y,z' (A, unstrained cell) when continuing a ramp (default: y0, z0)")
ap.add_argument("--moved", default="dd", choices=["dd", "dd_d1", "dd_s12", "tracker"],
                help="dd (Session 11 fix, default): moved = DD Burgers-triangle site differs from the start site (> 0.01 A); "
                     "tracker (Sessions 5-11 before the fix): energy-centroid shift >= one jump, which missed a one-site jump at "
                     "chi = -30/-24 (tracker shift 1.98 A)")
args = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
POT = f"{root}/potentials/Fe_mm.eam.fs"
os.makedirs("steps", exist_ok=True)
plog = open(f"{root}/logs/{args.name}.log", "a")
chi = np.radians(args.chi)


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True); plog.write(line + "\n"); plog.flush()


# ---- reference (unstrained) cell and surface slabs ----
L = open(args.ref or args.data).read().splitlines()
hdr = {k: [float(v) for v in l.split()[:2]] for l in L[:30] for k in ("xlo", "zlo") if l.rstrip().endswith(f"{k} {k[0]}hi")}
LX0, LZ0 = hdr["xlo"][1] - hdr["xlo"][0], hdr["zlo"][1] - hdr["zlo"][0]
XZ0 = float(next(l for l in L[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
n_atoms = int(next(l for l in L if l.strip().endswith("atoms")).split()[0])
i0 = L.index("Atoms # atomic") + 2
atoms = np.array([l.split()[:5] for l in L[i0:i0 + n_atoms]], float)
ids, y = atoms[:, 0].astype(int), atoms[:, 3]
d110 = A0 * np.sqrt(2) / 2
top, bot = ids[y > y.max() - 1.5 * d110], ids[y < y.min() + 1.5 * d110]
assert len(top) == len(bot)
with open("groups.lmp", "w") as f:
    f.write("group top id " + " ".join(map(str, top)) + "\ngroup bot id " + " ".join(map(str, bot)) + "\n")

# ---- loading model ----
if args.nocorr or args.model is None:
    Q = np.zeros((3, 4))
else:
    m = json.load(open(args.model))["correction"][str(int(round(args.chi)))]
    Q = np.array([m["q2"], m["q3"], m["q4"]])


def loading(tau):
    c = Q[0] * tau ** 2 + Q[1] * tau ** 3 + Q[2] * tau ** 4
    e = LIN * tau * np.sin(chi) + c[:3]
    return e, tau * np.cos(chi) + c[3]                # (eps_xx, eps_zz, gamma_xz), surface-force stress


say(f"{args.name}: chi = {args.chi:+.1f} deg (PROVISIONAL option-2 loading{', NO correction' if args.nocorr else ''}); "
    f"reference cell Lx = {LX0:.4f}, Lz = {LZ0:.4f}, xz = {XZ0:.6f} A, atoms = {n_atoms}, slabs {len(top)} + {len(bot)}")
for t in (1.0, 2.0):
    e, s = loading(t)
    say(f"loading check tau = {t}: sigma_xy target {t * np.cos(chi):.5f}, surface-force stress {s:.5f} GPa; "
        f"eps_xx {e[0]:+.7f} eps_zz {e[1]:+.7f} gamma_xz {e[2]:+.7f}")


def track(pe, yp, zp, Lz):
    out = subprocess.run([sys.executable, f"{root}/tools/track_core.py", pe, str(yp), str(zp), str(Lz)],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"y=(\S+) z=(\S+)", out)
    return float(m.group(1)), float(m.group(2))


def dd_site(data):
    """DD state of a relaxed step (tools/moved_rule.dd_state): triangle (dy, dz) relative to the inserted core, orientation, top column."""
    return dd_state(data, ".", args.nz, "steps/_dd_site.png")


def run_step(tau, datain, kind, start_core):
    tag = f"{kind}_{tau:.3f}"
    e, sxy = loading(tau)
    Lx, Lz = LX0 * (1 + e[0]), LZ0 * (1 + e[1])
    xz = XZ0 + e[2] * Lz
    fx = sxy * Lx * Lz * GPA_A2_TO_EV_A / len(top)
    ys, zs = start_core[0], start_core[1] * (1 + e[1])          # window centre in the strained cell
    # set the box for this tau (rebox.py: affine in x, z; y untouched) unless the input already has it
    H = open(datain).read(2000).splitlines()
    bx = float(next(l for l in H if l.rstrip().endswith("xlo xhi")).split()[1]); bz = float(next(l for l in H if l.rstrip().endswith("zlo zhi")).split()[1])
    bxz = float(next(l for l in H if l.rstrip().endswith("xy xz yz")).split()[1])
    if max(abs(bx - Lx), abs(bz - Lz), abs(bxz - xz)) > 1e-12:
        rebox(datain, f"steps/{tag}.in.data", Lx, Lz, xz); datain = f"steps/{tag}.in.data"
    cmd = ["mpirun", "-np", str(args.np), "lmp", "-in", f"{root}/tools/in.step_chi", "-screen", "none", "-log", f"steps/log.{tag}",
           "-var", "pot", POT, "-var", "datain", datain, "-var", "dataout", f"steps/{tag}.data", "-var", "peout", f"steps/{tag}.pe",
           "-var", "fx", f"{fx:.10f}", "-var", "y0", f"{args.y0}", "-var", "Lx", f"{Lx:.10f}", "-var", "Lz", f"{Lz:.10f}",
           "-var", "xz", f"{xz:.10f}", "-var", "ys", f"{ys:.6f}", "-var", "zs", f"{zs:.6f}", "-var", "maxit", str(args.maxit),
           "-var", "groups", "groups.lmp", "-var", "halt_dist", f"{HALT_JUMPS * JUMP:.4f}",
           "-var", "etol", args.etol, "-var", "ftol", args.ftol, "-var", "dmax", args.dmax, "-var", "maxeval", str(args.maxeval)]
    t = time.time()
    rc = subprocess.run(cmd).returncode
    log = open(f"steps/log.{tag}").read()
    if rc != 0 or "END " not in log:
        say(f"LAMMPS failed at tau = {tau:.3f} GPa (rc {rc})"); sys.exit(1)
    end = dict(kv.split("=") for kv in re.search(r"^END (.*)$", log, re.M).group(1).split())
    mstop = re.search(r"Stopping criterion = (.*)", log)
    stop = mstop.group(1).strip() if mstop else "halted"
    if "Fix halt condition" in log:
        stop = "halted (" + ("left 6 A window" if float(end["wfrac"]) < 0.4 else "band > 3 jumps") + ")"
    it = re.search(r"Iterations, force evaluations = (\d+) (\d+)", log)
    yc, zc = track(f"steps/{tag}.pe", ys, zs, Lz)
    zc_ref = zc / (1 + e[1])                                   # core z mapped back to the unstrained cell
    ds = dd_site(f"steps/{tag}.data")
    return dict(kind=kind, tau_GPa=tau, chi_deg=args.chi, sxy_target=tau * np.cos(chi), sxz_target=tau * np.sin(chi),
                sxy_surface=sxy, eps_xx=e[0], eps_zz=e[1], gamma_xz=e[2], E_total_eV=float(end["pe"]), E_pair_eV=float(end["epair"]),
                core_y_A=yc, core_z_ref_A=zc_ref % LZ0, band_dz_step_A=float(end["dcz"]), wfrac=float(end["wfrac"]),
                fmax=float(end["fmax"]), iterations=int(it.group(1)) if it else -1, stop=stop,
                converged="yes" if stop in ("energy tolerance", "force tolerance") else "no", fnorm=float(end["fnorm"]),
                seconds=round(time.time() - t, 1), data=f"steps/{tag}.data", core=(yc, zc_ref), dd_dy_A=ds["dy"], dd_dz_A=ds["dz"], dd_orient=ds["orient"],
                topcol_b=ds["topcol"], topcol_dy_A=ds["col"][0], topcol_dz_A=ds["col"][1])


rows, t0 = [], time.time()
lo, hi, st = map(float, args.coarse.split(":"))
coarse = [round(s, 4) for s in np.arange(lo, hi + st / 2, st)]
core0, dd0 = None, (tuple(map(float, args.dd0.split(","))) if args.dd0 else None)
core_events = []                                    # (tau, kind) of core-change events (D1 only)


def disp(r):
    dz = (r["core"][1] - core0[1] + LZ0 / 2) % LZ0 - LZ0 / 2
    return r["core"][0] - core0[0], dz


def record(r):
    rows.append(r)
    if core0 is not None:
        dy, dz = disp(r); r["dy_from_start_A"], r["dz_from_start_A"] = round(dy, 4), round(dz, 4)
        sdy, sdz = r["dd_dy_A"] - dd0[0], r["dd_dz_A"] - dd0[1]
        r["dd_shift_A"] = round(float(np.hypot(sdy, sdz)), 4)
        ev = classify(sdy, sdz, r["dd_orient"], r["topcol_b"])
        r["event"] = EVENT_NAME[ev]
        c = classify_d1(sdy, sdz, r["dd_orient"])
        r["core_change"] = "yes" if c == "core_change" else "no"
        if args.moved == "dd":
            r["moved"] = "yes" if is_moved(ev) else "no"
            if ev == "anomaly":
                say(f"ANOMALY at tau = {r['tau_GPa']:.3f} GPa: compact core shifted {r['dd_shift_A']:.3f} A (less than one jump); not moved (D5)")
        elif args.moved == "dd_d1":
            r["moved"] = "yes" if c == "moved" else "no"
            if c == "core_change":
                core_events.append((r["tau_GPa"], r["kind"]))
                say(f"CORE-CHANGE event at tau = {r['tau_GPa']:.3f} GPa: DD site shift {r['dd_shift_A']:.3f} A (orientation '{r['dd_orient']}'), "
                    f"less than one jump -> not moved (D1), ramp continues")
        elif args.moved == "dd_s12":
            r["moved"] = "yes" if np.hypot(sdy, sdz) > 0.01 else "no"
        else:
            r["moved"] = "yes" if np.hypot(dy, dz) >= JUMP else "no"
    say(f"step {len(rows)} [{r['kind']}] tau = {r['tau_GPa']:.3f} GPa (sxy {r['sxy_target']:.4f}, sxz {r['sxz_target']:+.4f}) | "
        f"core dy = {r.get('dy_from_start_A', 0):+.3f} dz = {r.get('dz_from_start_A', 0):+.3f} A | DD site ({r['dd_dy_A']:+.3f}, {r['dd_dz_A']:+.3f}, {r['dd_orient']}), "
        f"top column {r['topcol_b']:.3f} b {r.get('event', '')} "
        f"-> moved ({args.moved}): {r.get('moved', 'no')} | "
        f"band dz_step {r['band_dz_step_A']:+.3f} | converged: {r['converged']} ({r['stop']}, {r['iterations']} it, fmax {r['fmax']:.1e}) | "
        f"{r['seconds']} s | elapsed {time.time() - t0:.0f} s")


say(f"minimiser: CG, etol {args.etol}, ftol {args.ftol} (2-norm of the force vector), dmax {args.dmax} A, maxit {args.maxit}, maxeval {args.maxeval}, {args.np} MPI ranks")
say(f"ramp: coarse {coarse[0]}..{coarse[-1]} GPa step {st}, then {args.fine} GPa refinement; jump = {JUMP:.4f} A")
prev, prev_core = args.data, (tuple(map(float, args.start_core.split(","))) if args.start_core else (args.y0, args.z0))
last_unmoved, first_moved = None, None
for k, s in enumerate(coarse):
    r = run_step(s, prev, "coarse", prev_core)
    if core0 is None:
        core0 = prev_core if args.start_core else r["core"]
        if dd0 is None:
            dd0 = (r["dd_dy_A"], r["dd_dz_A"])
    record(r)
    if r["moved"] == "yes":
        first_moved = r; break
    last_unmoved = r
    prev, prev_core = r["data"], r["core"]
    say(f"progress: coarse step {k + 1} / {len(coarse)} (max) | elapsed {time.time() - t0:.0f} s")

if first_moved is not None and last_unmoved is not None:
    say(f"coarse bracket: unmoved {last_unmoved['tau_GPa']:.3f}, moved {first_moved['tau_GPa']:.3f} GPa -> refining")
    prev, prev_core = last_unmoved["data"], last_unmoved["core"]
    n = int(round((first_moved["tau_GPa"] - last_unmoved["tau_GPa"]) / args.fine))
    base = last_unmoved["tau_GPa"]                  # fixed: last unmoved COARSE stress (Session 11 fix: the loop used to add
    for j in range(1, n):                           # j*fine to the updated last_unmoved, giving +10, +20, +30 MPa steps)
        s = round(base + args.fine * j, 4)
        r = run_step(s, prev, "fine", prev_core); record(r)
        if r["moved"] == "yes":
            first_moved = r; break
        last_unmoved = r
        prev, prev_core = r["data"], r["core"]
    lu, fm = last_unmoved["tau_GPa"], first_moved["tau_GPa"]
    say(f"RESULT {args.name}: chi = {args.chi:+.1f} deg: Peierls stress (tau) between {lu:.3f} (last unmoved) and {fm:.3f} GPa (first moved)"
        f"{'; event: ' + first_moved.get('event', '') if args.moved == 'dd' else ''}")
    if core_events:
        say(f"CORE-CHANGE events (D1, not moves) at tau = " + ", ".join(f"{s:.3f} ({k})" for s, k in core_events) + " GPa")
    # DD maps at the bracket steps
    env = dict(os.environ, SCREW_NZ=str(args.nz))
    for lab, r in ((("last_unmoved", last_unmoved), ("first_moved", first_moved)) if args.ddmaps == "yes" else ()):
        out = subprocess.run([sys.executable, f"{root}/tools/ddmap_data.py", r["data"], f"{root}/figures/ddmap_{args.name}_{lab}.png",
                              f"{args.name}: tau = {r['tau_GPa']:.3f} GPa ({lab.replace('_', ' ')}), chi = {args.chi:+.0f} deg, provisional"],
                             capture_output=True, text=True, env=env).stdout
        for line in out.splitlines():
            say(f"DD {lab}: {line}")
else:
    say(f"RESULT {args.name}: no bracket found (first moved = {first_moved and first_moved['tau_GPa']}, last unmoved = {last_unmoved and last_unmoved['tau_GPa']})")

keys = ["kind", "chi_deg", "tau_GPa", "sxy_target", "sxz_target", "sxy_surface", "eps_xx", "eps_zz", "gamma_xz", "E_total_eV", "E_pair_eV",
        "core_y_A", "core_z_ref_A", "dy_from_start_A", "dz_from_start_A", "dd_dy_A", "dd_dz_A", "dd_orient", "dd_shift_A", "topcol_b", "topcol_dy_A", "topcol_dz_A", "event", "core_change", "moved", "band_dz_step_A", "wfrac", "converged", "stop",
        "iterations", "fmax", "fnorm", "seconds"]
with open(f"{root}/data/{args.name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
say(f"done: {len(rows)} steps, total {time.time() - t0:.0f} s; csv data/{args.name}.csv")
