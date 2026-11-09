"""Step 3 of V02_PLAN (Gate A-yz) driver, as defined in V02_PLAN section 8.9 (3 Nov 2026). Perfect slab, no dislocation.

For each of the 27 (chi, sigma_yz) pairs of jobs_step3.txt (chi -30, 0, +30 x sigma_yz -1.0 ... +1.0), one at a time:
  Run A (descriptive only, never a gate): the v0.1 strain model (checks/07_chi_calibration/strain_model.json, entry chi) with
    sigma_yz as a plain surface force (no correction); single measurements, no iteration, at tau = 0, 0.5, 1.0, 1.25, 1.5,
    1.75, 2.0, 2.2, 2.4 GPa (tau = 0 skipped at sigma_yz = 0: nothing loaded); per-tau verdict tools/gate_a.py gate_a3_check
    (section 8.8); summary: all pass or not, largest loading-angle error, largest other_pct.
  Run B (every pair): calibrate_yz.calibrate() with its own ladders (section 8.7) and rule (section 8.8), unchanged.
    A passing pair adds its entry "<chi>,<syz>" to strain_model_yz.json (as calibrate_yz.py's own main does).
After each pair: its LAMMPS logs and per-tau rows go to step3/chi<chi>_syz<syz>/, one line is appended to step3_results.csv
(the checkpoint), and a progress line is printed. Pairs already in step3_results.csv are skipped (resume).
At the end: per-pair table, s_A and the step-3 verdict (section 8.9):
  s_A = the largest s in {0.25, 0.5, 0.75, 1.0} such that Run B passes for every pair with 0 < |syz| <= s (all chi, both signs);
  step 3 PASS iff the three syz = 0 pairs pass Run B and s_A exists; otherwise FAIL.
Before running, the chi = -30, 0, +30 entries of checks/08_gateA_rebox/strain_model_al.json are compared with checks/07's.
Usage (from checks/10_gateA_yz; launch with tools/launch_safe.sh, NP from the launcher):
  python run_step3.py --dry-run      # list the 27 pairs and what would run; no LAMMPS
  python run_step3.py                # run (resumes from step3_results.csv)
"""
import argparse, csv, glob, json, os, shutil, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../tools"))
from calibrate_yz import calibrate, ladders, slab_stiffness, LIN, NAMES
from gate_a import gate_a3_check
from loading_v02 import model_correction

RUN_A_TAUS = (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.4)
S_LADDER = (0.25, 0.5, 0.75, 1.0)
MODEL07 = os.path.join(HERE, "../07_chi_calibration/strain_model.json")
MODEL08 = os.path.join(HERE, "../08_gateA_rebox/strain_model_al.json")
JOBS, RESULTS, STRAIN_MODEL, OUTDIR = "jobs_step3.txt", "step3_results.csv", "strain_model_yz.json", "step3"
FIELDS = ["chi", "syz", "runA_all_pass", "runA_n_pass", "runA_n", "runA_max_angle_deg", "runA_max_other_pct",
          "runB_verdict", "runB_n_pass", "runB_n", "runA_seconds", "runB_seconds", "finished"]


def read_jobs(path):
    out = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            c, s = line.split()
            out.append((float(c), float(s)))
    return out


def key(chi, syz):
    return f"{chi:g},{syz:+.2f}"


def al_identity(path07=MODEL07, path08=MODEL08, chis=(-30, 0, 30)):
    """{chi: True/False}: is the strain_model_al.json entry identical to checks/07's?"""
    a, b = json.load(open(path07))["correction"], json.load(open(path08))["correction"]
    return {c: a.get(str(c)) is not None and a.get(str(c)) == b.get(str(c)) for c in chis}


def run_a_loading(entry, chi, tau, syz):
    """Run A loading (section 8.9): strains and surface stress from a v0.1 strain-model entry at tau, sigma_yz plain (uncorrected).
    Returns (e(3) = eps_xx, eps_zz, gamma_xz; sigma_xy surface stress; sigma_yz surface stress)."""
    c = model_correction(entry, tau)
    e = LIN * tau * np.sin(np.radians(chi)) + c[:3]
    return e, tau * np.cos(np.radians(chi)) + c[3], syz


def run_a(chi, syz, entry, measure):
    rows = []
    for tau in RUN_A_TAUS:
        if tau == 0.0 and abs(syz) < 1e-12:
            continue
        e, sxy, sy = run_a_loading(entry, chi, tau, syz)
        s = measure(f"runA_chi{chi:+g}_syz{syz:+g}_tau{tau}", e, sxy, sy)
        g = gate_a3_check(s, chi, tau, syz)
        rows.append(dict(chi=chi, syz=syz, tau=tau, exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, syz_applied=sy,
                         **{f"meas_s{n}": v for n, v in zip(NAMES, s)}, angle_deg=float(np.degrees(g["ang"])),
                         other_pct=g["oth"] * 100, floor=g["floor"], verdict="PASS" if g["ok"] else "FAIL"))
    return rows


def step3_verdict(runb):
    """runb: {(chi, syz): True/False (Run B passed)} for all 27 pairs. Returns (s_A or None, zero_ok, 'PASS'/'FAIL'/'INCOMPLETE')."""
    pairs = [(c, s) for c in (-30.0, 0.0, 30.0) for s in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)]
    if any(p not in runb for p in pairs):
        return None, None, "INCOMPLETE"
    zero_ok = all(runb[(c, 0.0)] for c in (-30.0, 0.0, 30.0))
    s_a = None
    for s in S_LADDER:
        if all(runb[(c, y)] for (c, y) in pairs if 0 < abs(y) <= s + 1e-9):
            s_a = s
        else:
            break
    return s_a, zero_ok, "PASS" if (zero_ok and s_a is not None) else "FAIL"


def done_pairs(path):
    if not os.path.exists(path):
        return {}
    return {(float(r["chi"]), float(r["syz"])): r for r in csv.DictReader(open(path))}


def write_rows(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run_pairs(jobs, measure, K, results=RESULTS, outdir=OUTDIR, strain_model=STRAIN_MODEL, model07=MODEL07, say=print,
              calib=calibrate, move_logs=True):
    m07 = json.load(open(model07))["correction"]
    done = done_pairs(results)
    t0 = time.time()
    for i, (chi, syz) in enumerate(jobs, 1):
        if (chi, syz) in done:
            say(f"pair {i}/{len(jobs)} chi {chi:+g} syz {syz:+g}: already in {results}, skipped")
            continue
        pdir = os.path.join(outdir, f"chi{chi:+g}_syz{syz:+g}")
        os.makedirs(pdir, exist_ok=True)
        ta = time.time()
        a_rows = run_a(chi, syz, m07[str(int(chi))], measure)
        tb = time.time()
        b_rows, entry, b_pass = calib(chi, syz, measure, K, log=lambda m: None)
        te = time.time()
        write_rows(os.path.join(pdir, "runA.csv"), a_rows); write_rows(os.path.join(pdir, "runB.csv"), b_rows)
        if b_pass:
            m = json.load(open(strain_model)) if os.path.exists(strain_model) else {"correction": {}}
            m["correction"][key(chi, syz)] = entry
            m["note"] = "v0.2 step 3 (V02_PLAN 8.9): Run B calibrations of passing pairs (calibrate_yz.calibrate, ladders 8.7, rule 8.8)"
            json.dump(m, open(strain_model, "w"), indent=1)
        if move_logs:
            for f in glob.glob(f"log.runA_chi{chi:+g}_syz{syz:+g}_*") + glob.glob(f"log.yz_chi{chi:+g}_syz{syz:+g}_*"):
                shutil.move(f, os.path.join(pdir, f))
        a_ok = sum(r["verdict"] == "PASS" for r in a_rows)
        line = dict(chi=chi, syz=syz, runA_all_pass="yes" if a_ok == len(a_rows) else "no", runA_n_pass=a_ok, runA_n=len(a_rows),
                    runA_max_angle_deg=round(max(r["angle_deg"] for r in a_rows), 4), runA_max_other_pct=round(max(r["other_pct"] for r in a_rows), 3),
                    runB_verdict="PASS" if b_pass else "FAIL", runB_n_pass=sum(r["gate_A3"] == "PASS" for r in b_rows), runB_n=len(b_rows),
                    runA_seconds=round(tb - ta, 1), runB_seconds=round(te - tb, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        new = not os.path.exists(results)
        with open(results, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(line); f.flush(); os.fsync(f.fileno())
        say(f"pair {i}/{len(jobs)} chi {chi:+g} syz {syz:+g} | elapsed {time.time() - t0:.0f} s | Run A (descriptive) "
            f"{a_ok}/{len(a_rows)} pass, max angle {line['runA_max_angle_deg']:.4f} deg | Run B {line['runB_verdict']} "
            f"({line['runB_n_pass']}/{line['runB_n']})")
    return done_pairs(results)


def report(res, say=print):
    say("\nchi    syz   | Run A (descriptive): pass, max angle deg, max other % | Run B: verdict (rows)")
    for (chi, syz), r in sorted(res.items()):
        say(f"{chi:+4g} {syz:+6.2f} | {r['runA_n_pass']}/{r['runA_n']}, {float(r['runA_max_angle_deg']):.4f}, {float(r['runA_max_other_pct']):.3f}"
            f" | {r['runB_verdict']} ({r['runB_n_pass']}/{r['runB_n']})")
    s_a, zero_ok, v = step3_verdict({k: r["runB_verdict"] == "PASS" for k, r in res.items()})
    say(f"\nsigma_yz = 0 pairs pass Run B: {zero_ok}; s_A = {s_a}; STEP 3 (V02_PLAN 8.9): {v}")
    return s_a, zero_ok, v


def dry_run(jobs, say=print):
    ident = al_identity()
    say(f"strain_model_al.json vs checks/07 strain_model.json, chi -30 / 0 / +30 identical: {ident}")
    done = done_pairs(RESULTS)
    say(f"{len(jobs)} pairs ({len(set(jobs))} unique); already done: {len(done)}")
    for i, (chi, syz) in enumerate(jobs, 1):
        a_t = [t for t in RUN_A_TAUS if not (t == 0.0 and abs(syz) < 1e-12)]
        cal, chk = ladders(syz)
        state = "done (skip)" if (chi, syz) in done else "to run"
        say(f"pair {i:2d}/{len(jobs)} chi {chi:+4g} syz {syz:+5.2f} [{state}] | Run A: {len(a_t)} single measurements at tau {a_t} | "
            f"Run B: calibrate tau {list(cal)}, checks {list(chk)}")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(HERE)
    jobs = read_jobs(JOBS)
    if a.dry_run:
        dry_run(jobs); return 0
    ident = al_identity()
    print(f"strain_model_al.json vs checks/07, chi -30 / 0 / +30 identical: {ident}", flush=True)
    from calibrate_yz import lammps_measure
    measure = lammps_measure(os.environ.get("NP", "7"))
    res = run_pairs(jobs, measure, slab_stiffness(), say=lambda m: print(m, flush=True))
    s_a, zero_ok, v = report(res, say=lambda m: print(m, flush=True))
    return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
