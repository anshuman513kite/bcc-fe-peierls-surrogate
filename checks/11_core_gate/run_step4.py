"""Step 4 of V02_PLAN (core-must-not-move gate) driver, as defined in V02_PLAN section 8.10 (3 Nov 2026). Dislocation cell.

6 runs: chi -30, 0, +30 x sign (+, -). Per run, each state relaxed from the previous one (tools/in.step_chi, sweep settings
CG etol 0, ftol 1e-6, dmax 0.01 A):
  start (sigma_yz 0, tau 0) -> ladder sigma_yz 0.25, 0.5, 0.75 (x sign) at tau 0 -> tau 0.05 ... 0.6 at the top sigma_yz
  -> 2 stability seeds at the top state (tools/stability_test.py, its own settings, unchanged) -> unload tau 0.55 ... 0 at the
  top sigma_yz -> sigma_yz 0.5, 0.25, 0 at tau 0.
Loading at sigma_yz level L (every state, tau = 0 included): the step-3 Run B entry "<chi>,<L>" of
  checks/10_gateA_yz/strain_model_yz.json (5 components; tools/loading_v02.model_correction / syz_correction); sigma_xy and
  sigma_yz by surface forces (tools/loading_v02.surface_force, step_cmd); box strains through tools/rebox.py; slabs and
  reference box from sims/01_screw_cell/relaxed_down.data (the v0.1 sweep cell).
Checks per state (section 8.10):
  core: the existing core locator (tools/moved_rule.dd_state, the DD map finder): the Burgers-triangle centroid within 0.1 A of
    the start state's and the core not column-centred (top column < 1.4 b, D5 split threshold), i.e. no jump to another site;
  stability (top state, 2 seeds): stability_test.py "returned=yes";
  reversibility (last unload state): max |dr| < 0.01 A and |dE| < 1e-4 eV against the start state (same settings, zero load);
  non-affine (descriptive only, never a pass condition): largest per-atom residual of a least-squares affine fit of the
    displacement, (i) cumulative from the START state of the run (nonaffine_cum_A, drives the REVIEW flag, D-v02-5 a) and
    (ii) from the previous state (nonaffine_step_A).
Verdict: each state carries a level (start: 0; ladder state L < top: L; top ladder state, tau steps, stability and unload:
  the top level 0.75). s_4 = the largest level L in {0.25, 0.5, 0.75} such that every state with level <= L passes in all 6
  runs; s = min(s_A, s_4) with s_A = 0.75 (step 3). Step 4 PASS iff s_4 = 0.75 (the only level whose states include the full
  sequence: ladder, tau ramp, stability, unload). D-v02-5 (b): s_4 = 0.25 or 0.5 -> INCOMPLETE, a follow-up run of the full
  sequence at s_4 is required before s is used; no level -> FAIL.
D-v02-3 (Ansh, 3 Nov, before any step-4 data): the non-affine maximum is descriptive only; REVIEW flag: a run with any state
  whose cumulative non-affine maximum (from the start state, D-v02-5 a) exceeds 0.5 A is marked REVIEW and that state's data file is kept in
  checks/11_core_gate/review_snapshots/; the flag does not change s_4.
D-v02-4 (Ansh, 3 Nov): the choices above are frozen; logged per state (descriptive): the failed core condition ("site jump" /
  "split"), the final force norm, "linesearch alpha is zero" stops (count per run), at tau = 0 the off-axis stress measured for
  that entry in step 3 (Run B, perfect slab, tau = 0 calibrated row: |sxz|, |sxy| vs the Gate A floor; the dislocation-cell
  step itself measures no interior stress), and on the start state the max |dr| from relaxed_down.data. Stability kick seeds 1,
  2 are their own namespace (not surrogate / AL seeds).
Outputs: checks/11_core_gate/step4_results.csv (one line per state: the checkpoint), state data in
  sims/09_core_gate/<run>/steps/ (git-ignored, ~2.2 MB each), stability files in sims/09_core_gate/<run>/stab/.
Resume: states already in step4_results.csv are skipped; a run continues from its last finished state's data file.
Usage (from checks/11_core_gate; launch with tools/launch_safe.sh, NP from the launcher):
  python run_step4.py --dry-run        # list all states of the 6 runs and their loading entries; no LAMMPS
  python run_step4.py                  # run (resumes)
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from loading_v02 import surface_force, step_cmd, model_correction, syz_correction
from moved_rule import dd_state, SPLIT_B, JUMP
from render_lmp import render, cmd_vars, bad_numbers
from stability_args import make_parser as stability_parser

CHIS, SIGNS, LEVELS, TAUS = (-30, 0, 30), (+1, -1), (0.25, 0.5, 0.75), [round(0.05 * k, 2) for k in range(1, 13)]
S_A, CORE_TOL, REV_DR, REV_DE, N_SEEDS, REVIEW_A = 0.75, 0.1, 0.01, 1e-4, 2, 0.5
Y0, Z0, NZ = 74.0301, 72.2716, 21
CELL = os.path.join(ROOT, "sims/01_screw_cell/relaxed_down.data")
NPY = [os.path.join(ROOT, f"sims/01_screw_cell/screw_down.data.{k}.npy") for k in ("ref", "core")]
MODEL = os.path.join(ROOT, "checks/10_gateA_yz/strain_model_yz.json")
WORK = os.path.join(ROOT, "sims/09_core_gate")
RESULTS = os.path.join(HERE, "step4_results.csv")
SNAPDIR = os.path.join(HERE, "review_snapshots")
STEP3 = os.path.join(ROOT, "checks/10_gateA_yz/step3")
LIN = np.array([0.0, 0.0027625, 0.0164349])
MIN = dict(etol="0", ftol="1e-6", dmax="0.01", maxit=200000, maxeval=2000000)
FIELDS = ["run", "idx", "phase", "chi", "syz", "tau", "level", "model_key", "dd_dy", "dd_dz", "orient", "topcol_b", "shift_A",
          "core_ok", "fail_cond", "stab_returned", "rev_dr_A", "rev_dE_eV", "nonaffine_step_A", "nonaffine_cum_A", "review", "pe_eV", "fmax", "fnorm",
          "stop", "alpha_zero", "offaxis_tau0", "start_vs_cell_dr_A", "seconds", "pass", "data"]


def run_name(chi, sign):
    return f"chi{chi:+d}_{'p' if sign > 0 else 'm'}"


def model_key(chi, syz):
    return f"{chi:g},{syz:+.2f}"


def states(chi, sign):
    """Ordered states of one run: dicts with phase, syz, tau, level (section 8.10)."""
    top = LEVELS[-1] * sign
    out = [dict(phase="start", syz=0.0, tau=0.0, level=0.0)]
    out += [dict(phase="ladder", syz=L * sign, tau=0.0, level=L) for L in LEVELS]
    out += [dict(phase="load", syz=top, tau=t, level=LEVELS[-1]) for t in TAUS]
    out += [dict(phase=f"stab{k}", syz=top, tau=TAUS[-1], level=LEVELS[-1]) for k in range(1, N_SEEDS + 1)]
    out += [dict(phase="unload", syz=top, tau=t, level=LEVELS[-1]) for t in reversed([0.0] + TAUS[:-1])]
    out += [dict(phase="unload", syz=L * sign, tau=0.0, level=LEVELS[-1]) for L in reversed((0.0,) + LEVELS[:-1])]
    for i, s in enumerate(out):
        s["idx"] = i
        s["syz"] = s["syz"] or 0.0                           # -0.0 (unload of a negative run) -> 0.0, key "+0.00"
    return out


def entry_for(model, chi, syz):
    """The step-3 Run B entry for (chi, syz) (section 8.10); KeyError if missing."""
    return model["correction"][model_key(chi, syz)]


def loading(entry, chi, tau):
    c = model_correction(entry, tau)
    e = LIN * tau * np.sin(np.radians(chi)) + c[:3]
    return e, tau * np.cos(np.radians(chi)) + c[3], syz_correction(c)


def read_cell(path):
    L = open(path).read().splitlines()
    g = lambda k: [float(v) for v in next(l for l in L[:30] if l.rstrip().endswith(k)).split()[:2]]
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = g("xlo xhi"), g("ylo yhi"), g("zlo zhi")
    xz = float(next(l for l in L[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
    n = int(next(l for l in L if l.strip().endswith("atoms")).split()[0]); i0 = L.index("Atoms # atomic") + 2
    at = np.array([l.split()[:5] for l in L[i0:i0 + n]], float)
    at = at[np.argsort(at[:, 0])]
    return dict(Lx=xhi - xlo, Lz=zhi - zlo, xz=xz, ids=at[:, 0].astype(int), r=at[:, 2:5])


def displacement(a, b):
    """Displacement b - a per atom (same ids), minimum image along z (with the xz tilt) and x."""
    d = b["r"] - a["r"]
    n = np.round(d[:, 2] / b["Lz"]); d[:, 2] -= n * b["Lz"]; d[:, 0] -= n * b["xz"]
    d[:, 0] -= np.round(d[:, 0] / b["Lx"]) * b["Lx"]
    return d


def nonaffine_max(a, b):
    """Largest per-atom residual after a least-squares affine fit d = [r, 1] B of the displacement (descriptive)."""
    d = displacement(a, b)
    M = np.hstack([a["r"], np.ones((len(d), 1))])
    B = np.linalg.lstsq(M, d, rcond=None)[0]
    return float(np.linalg.norm(d - M @ B, axis=1).max())


def core_check(dd, dd0):
    """(shift, ok, failed condition): centroid within CORE_TOL of the start AND top column < SPLIT_B (D-v02-4 item 2)."""
    shift = float(np.hypot(dd["dy"] - dd0["dy"], dd["dz"] - dd0["dz"]))
    fails = [n for n, bad in (("site jump", shift > CORE_TOL), ("split", dd["topcol"] >= SPLIT_B)) if bad]
    return shift, not fails, "+".join(fails)


def offaxis_tau0(chi, syz, step3=STEP3):
    """Step-3 Run B measurement at tau = 0 for this entry (perfect slab): '|sxz| a, |sxy| b vs floor f GPa' (descriptive)."""
    if abs(syz) < 1e-12:
        return "n/a (syz = 0: no tau = 0 point)"
    path = os.path.join(step3, f"chi{chi:+g}_syz{syz:+g}", "runB.csv")
    r = next((r for r in csv.DictReader(open(path)) if r["kind"] == "calibrated" and float(r["tau"]) == 0.0), None)
    return "missing" if r is None else (f"|sxz| {abs(float(r['meas_sxz'])):.5f}, |sxy| {abs(float(r['meas_sxy'])):.5f} vs floor "
                                        f"{float(r['floor']):.6f} GPa (step 3)")


def run_summary(rows):
    """Per run: verdict (highest level whose states all pass), first failed state and condition, max non-affine, REVIEW."""
    out = {}
    for chi in CHIS:
        for sign in SIGNS:
            rn = run_name(chi, sign); rr = sorted([r for r in rows if r["run"] == rn], key=lambda r: int(r["idx"]))
            bad = [r for r in rr if r["pass"] not in ("yes", True)]
            lev = None
            for L in (0.0,) + LEVELS:
                if all(r["pass"] in ("yes", True) for r in rr if float(r["level"]) <= L + 1e-9) and rr:
                    lev = L
            na = [float(r["nonaffine_cum_A"]) for r in rr if r.get("nonaffine_cum_A") not in (None, "")]
            out[rn] = dict(states=len(rr), highest_level_passed=(lev if lev else None),
                           first_fail=(f"{bad[0]['phase']} syz {float(bad[0]['syz']):+.2f} tau {float(bad[0]['tau']):.2f}: "
                                       f"{bad[0].get('fail_cond') or ('stability' if str(bad[0]['phase']).startswith('stab') else 'reversibility')}") if bad else "",
                           max_nonaffine_A=max(na) if na else None, review="REVIEW" if any(r.get("review") == "yes" for r in rr) else "",
                           alpha_zero_stops=sum(r.get("alpha_zero") == "yes" for r in rr))
    return out


def verdict(rows):
    """rows: result dicts (all states of all runs). Returns (s_4 or None, s or None, 'PASS'/'FAIL'/'INCOMPLETE', note).
    s_4 nested over levels; PASS only if s_4 = 0.75 (full sequence faced); s_4 = 0.25 / 0.5 -> INCOMPLETE, follow-up run of
    the full sequence at s_4 required before s is used (D-v02-5 b); no level -> FAIL; missing states -> INCOMPLETE."""
    need = {(run_name(c, g), s["idx"]) for c in CHIS for g in SIGNS for s in states(c, g)}
    have = {(r["run"], int(r["idx"])): r for r in rows}
    if not need <= set(have):
        return None, None, "INCOMPLETE", f"{len(need - set(have))} states missing"
    s4 = None
    for L in LEVELS:
        if all(have[k]["pass"] in ("yes", True) for k in need if float(have[k]["level"]) <= L + 1e-9):
            s4 = L
        else:
            break
    if s4 is None:
        return None, None, "FAIL", "no level passes in all 6 runs"
    if s4 < LEVELS[-1]:
        return s4, None, "INCOMPLETE", (f"follow-up at {s4}: the full sequence (ladder, tau 0 -> 0.6, 2 stability seeds, unload) "
                                        f"at sigma_yz = +-{s4} in all 6 runs is required before s is used (D-v02-5 b)")
    return s4, min(S_A, s4), "PASS", ""


def step_numbers(entry, chi, tau, syz, LX0, LZ0, XZ0, n_slab):
    """Box and surface forces of one state: (Lx, Lz, xz, fx, fz or None, e). Shared by the run and the dry-run render check."""
    e, sxy, dsyz = loading(entry, chi, tau)
    Lx, Lz = LX0 * (1 + e[0]), LZ0 * (1 + e[1]); xz = XZ0 + e[2] * Lz
    fx = surface_force(sxy, Lx, Lz, n_slab)
    fz = surface_force(syz + dsyz, Lx, Lz, n_slab) if syz != 0.0 else None
    return Lx, Lz, xz, fx, fz, e


def stab_cmd_vars(Lx, Lz, xz, fx, fz, seed):
    """The -var values tools/stability_test.py passes for a state (same formulas and formats as that script)."""
    v = dict(fx=f"{fx:.10f}", y0=f"{Y0}", Lx=f"{Lx:.10f}", Lz=f"{Lz:.10f}", xz=f"{xz:.10f}", ys=f"{Y0:.6f}", zs=f"{Z0:.6f}",
             maxit="200000", halt_dist=f"{3 * JUMP:.4f}", etol="0", ftol="1e-6", dmax="0.01", maxeval="2000000",
             jiggle="0.001", seed=str(seed))
    if fz is not None:
        v.update(fz=f"{fz:.10f}", fzb=f"{-fz:.10f}")
    if fx < 0:
        v["fxb"] = f"{-fx:.10f}"
    return v


def render_check(model, say=print):
    """Render tools/in.step_chi for ALL states of the 6 runs (relaxations through loading_v02.step_cmd, stability seeds with the
    stability_test.py variables) and check every numeric parameter parses (no '--', '+-', nan, inf). Returns (states, problems)."""
    ref = open(CELL).read().splitlines()
    g = lambda k: [float(v) for v in next(l for l in ref[:30] if l.rstrip().endswith(k)).split()[:2]]
    LX0, LZ0 = g("xlo xhi")[1] - g("xlo xhi")[0], g("zlo zhi")[1] - g("zlo zhi")[0]
    XZ0 = float(next(l for l in ref[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
    c = read_cell(CELL); y = c["r"][:, 1]; n_slab = int((y > y.max() - 1.5 * 2.8553 * np.sqrt(2) / 2).sum())
    tmpl, n, problems = os.path.join(ROOT, "tools/in.step_chi"), 0, []
    for chi in CHIS:
        for sign in SIGNS:
            for s in states(chi, sign):
                Lx, Lz, xz, fx, fz, e = step_numbers(entry_for(model, chi, s["syz"]), chi, s["tau"], s["syz"], LX0, LZ0, XZ0, n_slab)
                if s["phase"].startswith("stab"):
                    v = dict(stab_cmd_vars(Lx, Lz, xz, fx, fz, int(s["phase"][-1])), pot="P", datain="in.data", dataout="out.data",
                             peout="out.pe", groups="g.lmp")
                    cmd = [x for k, val in v.items() for x in ("-var", k, val)]
                else:
                    cmd = step_cmd(7, tmpl, "log", "P", "in.data", "out.data", "out.pe", fx, Y0, Lx, Lz, xz, Y0, Z0 * (1 + e[1]),
                                   MIN["maxit"], "g.lmp", 3 * JUMP, MIN["etol"], MIN["ftol"], MIN["dmax"], MIN["maxeval"], fz=fz)
                    v = cmd_vars(cmd)
                bad = bad_numbers(render(tmpl, v), cmd)
                n += 1
                problems += [f"{run_name(chi, sign)} state {s['idx']} ({s['phase']}): {b}" for b in bad]
    return n, problems


def stab_argv(data, tau, chi, tag, seed, np_ranks, key, syz, nz=NZ, cell=CELL, model_path=MODEL):
    """argv of tools/stability_test.py for one stability seed. Every option whose value can start with '-' is passed as
    --opt=value (3 Nov: '--model_key -30,+0.75' was read by argparse as an option and the helper crashed).
    model_path: the strain-model file (default the step-3 one; 4 Nov: production passes its own, V02_PLAN 8.12 note d)."""
    return [data, f"{tau}", f"{chi}", f"{Y0}", tag, f"--seed={seed}", f"--np={np_ranks}", f"--nz={nz}", f"--model={model_path}",
            f"--model_key={key}", f"--syz={syz}", f"--ref={cell}"]


def stab_parse_check(work=WORK):
    """Parse the exact argv of all 12 stability seeds (6 runs x 2) with stability_test.py's own parser (no LAMMPS). Returns
    (n, problems)."""
    n, problems = 0, []
    for chi in CHIS:
        for sign in SIGNS:
            rn = run_name(chi, sign)
            for s in states(chi, sign):
                if not s["phase"].startswith("stab"):
                    continue
                argv = stab_argv(os.path.join(work, rn, "steps/top.data"), s["tau"], chi, f"{rn}_{s['phase']}", int(s["phase"][-1]), 7,
                                 model_key(chi, s["syz"]), s["syz"])
                n += 1
                try:
                    a = stability_parser().parse_args(argv)
                    if a.model_key != model_key(chi, s["syz"]) or a.syz != s["syz"] or a.chi != chi or a.seed != int(s["phase"][-1]):
                        problems.append(f"{rn} {s['phase']}: parsed values differ: {vars(a)}")
                except SystemExit:
                    problems.append(f"{rn} {s['phase']}: argparse rejected {argv}")
    return n, problems


class LammpsRunner:
    """One relaxed state through tools/in.step_chi (same command as tools/peierls_chi.py run_step, via loading_v02.step_cmd).
    cell / npy / nz / z0 default to the 146.87 A cell of step 4; steps 5-6 (3 Nov) pass the 2x cell for step 6."""

    def __init__(self, np_ranks, cell=CELL, npy=NPY, nz=NZ, z0=Z0, model_path=MODEL):
        self.np, self.cell, self.npy, self.nz, self.z0, self.model_path = np_ranks, cell, npy, nz, z0, model_path
        ref = open(cell).read().splitlines()
        g = lambda k: [float(v) for v in next(l for l in ref[:30] if l.rstrip().endswith(k)).split()[:2]]
        self.LX0, self.LZ0 = g("xlo xhi")[1] - g("xlo xhi")[0], g("zlo zhi")[1] - g("zlo zhi")[0]
        self.XZ0 = float(next(l for l in ref[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
        c = read_cell(cell); y = c["r"][:, 1]; d110 = 2.8553 * np.sqrt(2) / 2
        self.top, self.bot = c["ids"][y > y.max() - 1.5 * d110], c["ids"][y < y.min() + 1.5 * d110]

    def setup(self, rdir):
        os.makedirs(os.path.join(rdir, "steps"), exist_ok=True)
        for f in self.npy:
            if not os.path.exists(os.path.join(rdir, os.path.basename(f))):
                shutil.copy(f, rdir)
        with open(os.path.join(rdir, "groups.lmp"), "w") as f:
            f.write("group top id " + " ".join(map(str, self.top)) + "\ngroup bot id " + " ".join(map(str, self.bot)) + "\n")

    def relax(self, rdir, tag, datain, entry, chi, tau, syz):
        from rebox import rebox
        Lx, Lz, xz, fx, fz, e = step_numbers(entry, chi, tau, syz, self.LX0, self.LZ0, self.XZ0, len(self.top))
        H = open(datain).read(2000).splitlines()
        bx = float(next(l for l in H if l.rstrip().endswith("xlo xhi")).split()[1]); bz = float(next(l for l in H if l.rstrip().endswith("zlo zhi")).split()[1])
        bxz = float(next(l for l in H if l.rstrip().endswith("xy xz yz")).split()[1])
        if max(abs(bx - Lx), abs(bz - Lz), abs(bxz - xz)) > 1e-12:
            rebox(datain, os.path.join(rdir, f"steps/{tag}.in.data"), Lx, Lz, xz); datain = f"steps/{tag}.in.data"
        cmd = step_cmd(self.np, f"{ROOT}/tools/in.step_chi", f"steps/log.{tag}", f"{ROOT}/potentials/Fe_mm.eam.fs", datain, f"steps/{tag}.data",
                       f"steps/{tag}.pe", fx, Y0, Lx, Lz, xz, Y0, self.z0 * (1 + e[1]), MIN["maxit"], "groups.lmp", 3 * JUMP,
                       MIN["etol"], MIN["ftol"], MIN["dmax"], MIN["maxeval"], fz=fz)
        rc = subprocess.run(cmd, cwd=rdir).returncode
        log = open(os.path.join(rdir, f"steps/log.{tag}")).read()
        if rc != 0 or "END " not in log:
            err = [l for l in log.splitlines() if l.startswith("ERROR")]
            raise RuntimeError(f"LAMMPS failed for {tag} (rc {rc}): {err[-1] if err else 'no END line'}")
        end = dict(kv.split("=") for kv in re.search(r"^END (.*)$", log, re.M).group(1).split())
        m = re.search(r"Stopping criterion = (.*)", log)
        stop = "halted" if "Fix halt condition" in log else (m.group(1).strip() if m else "?")
        out = os.path.join(rdir, f"steps/{tag}.data")
        return dict(data=out, pe=float(end["pe"]), fmax=float(end["fmax"]), fnorm=float(end["fnorm"]), stop=stop,
                    dd=dd_state(out, rdir, self.nz, os.path.join(rdir, "steps/_dd.png")))

    def stability(self, rdir, tag, data, entry_key, chi, tau, syz, seed):
        cmd = [sys.executable, f"{ROOT}/tools/stability_test.py"] + stab_argv(data, tau, chi, tag, seed, self.np, entry_key, syz,
                                                                            self.nz, self.cell, self.model_path)
        p = subprocess.run(cmd, cwd=rdir, capture_output=True, text=True)
        if p.returncode != 0:                          # 3 Nov: a helper crash stops the driver; it is never a failed check
            raise RuntimeError(f"stability_test.py crashed for {tag} (rc {p.returncode}): {p.stderr.strip()[-500:]}")
        m = re.search(r"returned=(yes|no)", p.stdout)
        if not m:
            raise RuntimeError(f"stability_test.py gave no 'returned=' line for {tag}: {p.stdout.strip()[-300:]}")
        return m.group(1) == "yes", p.stdout.strip().splitlines()[-1]


def done_rows(path=RESULTS):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def run_all(runner, model, results=RESULTS, work=WORK, say=print, read=read_cell, snapdir=SNAPDIR, offaxis=offaxis_tau0):
    rows = done_rows(results); done = {(r["run"], int(r["idx"])): r for r in rows}
    t0 = time.time()
    for chi in CHIS:
        for sign in SIGNS:
            rn = run_name(chi, sign); rdir = os.path.join(work, rn); runner.setup(rdir)
            st = states(chi, sign)
            cur, dd0, start_cell, prev_cell = CELL, None, None, None
            for s in st:
                key = (rn, s["idx"])
                if key in done:                                      # resume: restore the run's state from the checkpoint
                    r = done[key]
                    if not s["phase"].startswith("stab"):
                        cur = r["data"]
                    if s["phase"] == "start":
                        dd0 = dict(dy=float(r["dd_dy"]), dz=float(r["dd_dz"])); start_cell = (read(r["data"]), float(r["pe_eV"]))
                    continue
                k = model_key(chi, s["syz"])
                entry = entry_for(model, chi, s["syz"])
                ts = time.time()
                row = dict(run=rn, idx=s["idx"], phase=s["phase"], chi=chi, syz=s["syz"], tau=s["tau"], level=s["level"], model_key=k)
                if s["phase"].startswith("stab"):
                    ok, line = runner.stability(rdir, f"{rn}_{s['phase']}", cur, k, chi, s["tau"], s["syz"], int(s["phase"][-1]))
                    row.update(stab_returned="yes" if ok else "no", stop=line[:120], **{"pass": "yes" if ok else "no"})
                else:
                    tag = f"{s['idx']:02d}_{s['phase']}_syz{s['syz']:+.2f}_tau{s['tau']:.2f}"
                    res = runner.relax(rdir, tag, cur, entry, chi, s["tau"], s["syz"])
                    cell = read(res["data"])
                    if s["phase"] == "start":
                        dd0, start_cell = res["dd"], (cell, res["pe"])
                    if prev_cell is None:                            # first relaxation of this session: compare with its input
                        prev_cell = read(cur)
                    shift, core_ok, cond = core_check(res["dd"], dd0)
                    ok = core_ok
                    na_step = nonaffine_max(prev_cell, cell)
                    na = nonaffine_max(start_cell[0], cell)               # cumulative from the start state (D-v02-5 a)
                    row.update(dd_dy=res["dd"]["dy"], dd_dz=res["dd"]["dz"], orient=res["dd"]["orient"], topcol_b=res["dd"]["topcol"],
                               shift_A=round(shift, 4), core_ok="yes" if core_ok else "no", fail_cond=cond, nonaffine_step_A=round(na_step, 4),
                               nonaffine_cum_A=round(na, 4),
                               review="yes" if na > REVIEW_A else "no", pe_eV=res["pe"], fmax=res["fmax"], fnorm=res.get("fnorm", ""),
                               stop=res["stop"], alpha_zero="yes" if "alpha" in str(res["stop"]) else "no", data=res["data"])
                    if s["tau"] == 0.0:
                        row["offaxis_tau0"] = offaxis(chi, s["syz"])
                    if s["phase"] == "start":
                        row["start_vs_cell_dr_A"] = round(float(np.linalg.norm(displacement(read(CELL), cell), axis=1).max()), 5)
                    if na > REVIEW_A:                                 # D-v02-3(b), D-v02-5(a): keep the snapshot; verdict unchanged
                        os.makedirs(snapdir, exist_ok=True)
                        shutil.copy(res["data"], os.path.join(snapdir, f"{rn}_{os.path.basename(res['data'])}"))
                    if s["idx"] == st[-1]["idx"]:                     # last unload state: reversibility against the start state
                        dr = float(np.linalg.norm(displacement(start_cell[0], cell), axis=1).max()); dE = abs(res["pe"] - start_cell[1])
                        row.update(rev_dr_A=dr, rev_dE_eV=dE); ok = ok and dr < REV_DR and dE < REV_DE
                        if not (dr < REV_DR and dE < REV_DE):
                            row["fail_cond"] = "+".join(filter(None, [row["fail_cond"], "reversibility"]))
                    row["pass"] = "yes" if ok else "no"
                    cur, prev_cell = res["data"], cell
                row["seconds"] = round(time.time() - ts, 1)
                append(results, row)
                say(f"{rn} state {s['idx'] + 1}/{len(st)} {s['phase']} syz {s['syz']:+.2f} tau {s['tau']:.2f} | pass {row['pass']}"
                    f"{' shift ' + str(row.get('shift_A')) + ' A' if 'shift_A' in row else ''} | elapsed {time.time() - t0:.0f} s")
    return done_rows(results)


def dry_run(model, say=print):
    n, problems = render_check(model)
    say(f"render check: tools/in.step_chi rendered for {n} states; numeric parameters not parsing: {len(problems)}")
    for p in problems[:20]:
        say("   BAD " + p)
    ns, sp = stab_parse_check()
    say(f"stability argv check: {ns} commands parsed with stability_test.py's parser; problems: {len(sp)}")
    for p in sp:
        say("   BAD " + p)
    done = {(r["run"], int(r["idx"])) for r in done_rows()}
    for chi in CHIS:
        for sign in SIGNS:
            rn, st = run_name(chi, sign), states(chi, sign)
            missing = sorted({model_key(chi, s["syz"]) for s in st if model_key(chi, s["syz"]) not in model["correction"]})
            say(f"{rn}: {len(st)} states ({sum(not s['phase'].startswith('stab') for s in st)} relaxations + {N_SEEDS} stability), "
                f"done {sum((rn, s['idx']) in done for s in st)}; step-3 entries missing: {missing or 'none'}")
            for s in st:
                say(f"   {s['idx']:2d} {s['phase']:7s} syz {s['syz']:+.2f} tau {s['tau']:.2f} level {s['level']:.2f} entry {model_key(chi, s['syz'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    model = json.load(open(MODEL))
    if a.dry_run:
        dry_run(model)
        return 1 if (render_check(model)[1] or stab_parse_check()[1]) else 0
    n, problems = render_check(model)
    problems += stab_parse_check()[1]
    if problems:
        print(f"pre-run checks FAILED ({len(problems)} problems); not running. First: {problems[0]}", flush=True)
        return 2
    rows = run_all(LammpsRunner(os.environ.get("NP", "7")), model, say=lambda m: print(m, flush=True))
    print("\nrun        states  highest level passed  first failed state (condition)            max cum. non-affine A  REVIEW  alpha-zero", flush=True)
    for rn, r in run_summary(rows).items():
        print(f"{rn:10s} {r['states']:6d}  {str(r['highest_level_passed']):20s}  {r['first_fail'] or '-':42s} "
              f"{r['max_nonaffine_A'] if r['max_nonaffine_A'] is not None else '-':>16}  {r['review'] or '-':6s}  {r['alpha_zero_stops']}", flush=True)
    s4, s, v, note = verdict(rows)
    print(f"s_4 = {s4}; s = {s}; STEP 4 (V02_PLAN 8.10, D-v02-5): {v}{'; ' + note if note else ''}", flush=True)
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
