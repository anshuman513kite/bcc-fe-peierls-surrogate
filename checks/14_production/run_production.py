"""Production driver for night T and night P (V02_PLAN section 8.12, D-v02-8; 4 Nov 2026).

Per point, in job order:
  1. calibration: every pair the point needs (its target and its pre-ramp levels, note a) that has no calibration record gets
     its own checks/10_gateA_yz calibrate_yz.calibrate() run (unchanged code; step-3 Run B criteria, section 8.8 floor; item 10).
     Every verdict goes to calibrations.csv (the checkpoint); a passing entry goes to strain_model_prod.json (note d).
  2. Gate A verdict: a point with any failed pair is INVALID ("gate_a_fail"), no ramp.
  3. render + parse every possible LAMMPS input and stability argv of the point (ramp_v02.pre_run_checks with the real
     entries); a problem stops the driver (exit 2), it is never scored.
  4. ramp: ramp_v02.run_ramp (start, pre-ramp levels, coarse 0.05 to 3.0, fine 0.01, stability seeds 1 and 2), one CSV line
     per state (T: data/sealed/T_states.csv; P: data/production/P_states.csv); a helper or LAMMPS crash raises.
  5. acceptance (item 8, note c): invalid if no bracket (incl. a move during the pre-ramp), anomaly, or a failed seed; the
     below-top check is recorded only. One row in the results (T: data/sealed/T_results.csv; P: data/production/P_results.csv).
  6. stop rule (item 8): > 2 invalid of T or > 18 invalid of P -> STOP for Ansh (exit 3), checked after every point and at start.
Modes: --set T (the 49 T points; the 9 step-5 nodes are already in the sealed results); --set P --night N (the batch of night N:
the first 50 P points not done in the permutation order, fixed in data/production/P_night<N>.json at the first launch).
Resume: start the same command again; points with a result row are skipped, a ramp resumes from its state CSV.
T is blind (D-v02-9 item 3): for --set T the progress lines and the final summary show counts and reasons only (no tau_P,
bracket or event values; the per-state ramp lines are not printed); the values go only to the sealed CSVs.
Exit: 0 = the set / night batch is complete; 2 = pre-run check problem; 3 = invalid-count stop; a crash raises (exit 1).
Usage (from checks/14_production; launch with tools/launch_safe.sh, NP from the launcher):
  python run_production.py --set T [--dry-run]
  python run_production.py --set P --night 1 [--dry-run]
"""
import argparse, csv, glob, json, os, shutil, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
for p in ("tools", "checks/10_gateA_yz", "checks/11_core_gate", "checks/12_corner_gate"):
    sys.path.insert(0, os.path.join(ROOT, p))
import make_jobs_v02 as mj
import ramp_v02 as rp
r4 = rp.r4

STEP3_MODEL = r4.MODEL
PROD_MODEL = os.path.join(HERE, "strain_model_prod.json")
CALIB_CSV, CALIB_DIR = os.path.join(HERE, "calibrations.csv"), os.path.join(HERE, "calib")
CALIB_TMP = os.path.join(HERE, "calib_tmp")   # D-v02-10 log-file note: calibrator logs and temporary data go here, not to GATE_DIR
GATE_DIR = os.path.join(ROOT, "checks/10_gateA_yz")
RESULTS = dict(T=mj.SEALED_T, P=os.path.join(mj.PROD, "P_results.csv"))
STATES = dict(T=os.path.join(os.path.dirname(mj.SEALED_T), "T_states.csv"), P=os.path.join(mj.PROD, "P_states.csv"))
WORK = dict(T=os.path.join(ROOT, "sims/12_production/T"), P=os.path.join(ROOT, "sims/12_production/P"))
MAX_INVALID = dict(T=2, P=18)                          # item 8: stop if MORE than this many are invalid
NIGHT_CAP = 50
ZERO_ENTRY = {f"q{p}": [0.0] * 5 for p in (2, 3, 4)}  # note b: what every sigma_yz = 0 entry gives at tau = 0
# time estimates (dry run): measured step-5 ramp per point (checks/12_corner_gate/step5_results.csv, 9 runs, 7 ranks) and
# step-3 Run B calibration per pair (checks/10_gateA_yz/step3_results.csv, 24 pairs with sigma_yz != 0)
RAMP_S, RAMP_S_MAX, CALIB_S, CALIB_S_MAX = 545, 704, 24.2, 30.6
CALIB_FIELDS = ["model_key", "chi", "syz", "verdict", "n_pass", "n", "seconds", "for_point", "finished"]


def read_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def append_row(path, row, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def init_model(path=PROD_MODEL, step3=STEP3_MODEL):
    """Production strain model (note d): the step-3 entries copied unchanged, then the production entries."""
    if not os.path.exists(path):
        m = json.load(open(step3))
        json.dump(dict(correction=dict(m["correction"]), note="v0.2 production (V02_PLAN 8.12 note d): the 26 step-3 entries "
                       "(checks/10_gateA_yz/strain_model_yz.json, unchanged) + one calibrate_yz entry per new (chi, syz) pair"),
                  open(path, "w"), indent=1)
    return json.load(open(path))


def entry_fn_for(target_syz):
    """Entry lookup for one point: the real entry; at sigma_yz = 0 with no entry and a non-zero target, the zero entry (note b:
    the start state at tau = 0, where every sigma_yz = 0 entry gives exactly zero). Anything else missing raises KeyError."""
    def fn(model, chi, syz):
        k = r4.model_key(chi, syz)
        if k in model["correction"]:
            return model["correction"][k]
        if syz == 0 and target_syz != 0:
            return ZERO_ENTRY
        raise KeyError(k)
    return fn


def pairs_needed(job):
    """(chi, syz) pairs the point needs calibrated: its pre-ramp levels and its target (note a)."""
    out = [(job["chi"], lv) for lv in job["levels"]]
    return out if job["syz"] != 0 else [(job["chi"], 0.0)]


def calib_status(model, calib_rows):
    """{model_key: 'PASS'/'FAIL'}: step-3 entries count as PASS (step 3 Run B), plus every production verdict."""
    st = {k: "PASS" for k in model["correction"]}
    st.update({r["model_key"]: r["verdict"] for r in calib_rows})
    return st


def acceptance(summ):
    """Item 8 / note c: (valid, reasons). The below-top check is recorded, never a reason."""
    c, reasons = summ["checks"], []
    if not c["bracket"] or not c["no_preramp_move"]:
        reasons.append("no_bracket" + ("" if c["no_preramp_move"] else " (moved during pre-ramp)"))
    elif not c["first_event_ok"]:
        reasons.append(f"anomaly (first event {summ['first_event']})")
    if c["bracket"] and not c["stability"]:
        reasons.append("failed_seed")
    return not reasons, reasons


def n_invalid(results):
    return sum(r["valid"] == "no" for r in read_rows(results))


def stop_reached(set_, n_bad):
    return n_bad > MAX_INVALID[set_]


def night_batch(night, p_jobs, done_ids, prod=mj.PROD, write=True):
    """Night N's batch (note e): fixed in P_night<N>.json at the first launch; else the first NIGHT_CAP not-done points in the
    permutation order."""
    path = os.path.join(prod, f"P_night{night}.json")
    if os.path.exists(path):
        return json.load(open(path))["ids"]
    ids = [j["id"] for j in sorted(p_jobs, key=lambda j: j["perm_rank"]) if j["id"] not in done_ids][:NIGHT_CAP]
    if write:
        os.makedirs(prod, exist_ok=True)
        json.dump(dict(night=night, ids=ids, created=time.strftime("%Y-%m-%d %H:%M:%S")), open(path, "w"), indent=1)
    return ids


def run_name(job):
    return f"{job['id']}_chi{job['chi']:+.4f}_syz{job['syz']:+.5f}"


class LammpsCalibrator:
    """calibrate_yz.calibrate() on the Gate A slab of checks/10_gateA_yz (its relative inputs need that cwd); logs moved to
    calib/<key>/ afterwards, as run_step3.py does. Its LAMMPS logs (incl. the reference log.yz_ref) and temporary data files are
    written to CALIB_TMP, never into checks/10_gateA_yz (D-v02-10 log-file note)."""

    def __init__(self, np_ranks):
        self.np, self.measure = np_ranks, None

    def __call__(self, chi, syz, tag):
        import calibrate_yz as cy
        os.makedirs(CALIB_TMP, exist_ok=True)
        cwd = os.getcwd(); os.chdir(GATE_DIR)
        try:
            if self.measure is None:
                self.measure, self.K = cy.lammps_measure(self.np, out_dir=CALIB_TMP), cy.slab_stiffness()
            rows, entry, ok = cy.calibrate(chi, syz, self.measure, self.K, log=lambda m: None)
            d = os.path.join(CALIB_DIR, tag); os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(CALIB_TMP, f"log.yz_chi{chi:+g}_syz{syz:+g}_*")):
                shutil.move(f, os.path.join(d, os.path.basename(f)))
        finally:
            os.chdir(cwd)
        with open(os.path.join(d, "runB.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        return rows, entry, ok


def calibrate_pairs(job, model, calibrate, say, model_path=PROD_MODEL):
    """Calibrate the pairs of job that have no record. Returns (failed keys, number of new calibrations, seconds)."""
    st, n, t0 = calib_status(model, read_rows(CALIB_CSV)), 0, time.time()
    for chi, syz in pairs_needed(job):
        k = r4.model_key(chi, syz)
        if k in st:
            continue
        ts = time.time()
        rows, entry, ok = calibrate(chi, syz, k.replace(",", "_"))
        n_pass = sum(r["gate_A3"] == "PASS" for r in rows)
        if ok:
            model["correction"][k] = entry
            json.dump(model, open(model_path + ".tmp", "w"), indent=1); os.replace(model_path + ".tmp", model_path)
        append_row(CALIB_CSV, dict(model_key=k, chi=chi, syz=syz, verdict="PASS" if ok else "FAIL", n_pass=n_pass, n=len(rows),
                                   seconds=round(time.time() - ts, 1), for_point=job["id"], finished=time.strftime("%Y-%m-%d %H:%M:%S")),
                   CALIB_FIELDS)
        st[k] = "PASS" if ok else "FAIL"; n += 1
        say(f"   calibration {k}: Gate A {st[k]} ({n_pass}/{len(rows)}) | {time.time() - ts:.0f} s")
    failed = [r4.model_key(c, s) for c, s in pairs_needed(job) if st[r4.model_key(c, s)] != "PASS"]
    return failed, n, round(time.time() - t0, 1)


BLIND = {"T"}                                          # D-v02-9 item 3: sets whose values are not shown (counts and reasons only)


def shown_reason(set_, reason):
    """The invalid reason as printed: for a blind set the event in "anomaly (first event X)" is dropped (D-v02-9 item 3)."""
    if set_ not in BLIND:
        return reason
    return ";".join("anomaly" if r.startswith("anomaly") else r for r in reason.split(";"))


def counts_summary(set_, results):
    """Counts only (D-v02-9 item 3): points done / valid / invalid, and the invalid reasons (event values dropped for T)."""
    rows = read_rows(results)
    bad = [r for r in rows if r["valid"] == "no"]
    reasons = {}
    for r in bad:
        for x in shown_reason(set_, r["reason"]).split(";"):
            k = x.split(":")[0]
            reasons[k] = reasons.get(k, 0) + 1
    rs = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "none"
    return f"{set_} results: {len(rows)} done, {len(rows) - len(bad)} valid, {len(bad)} invalid (reasons: {rs})"


def run_points(set_, jobs, runner, calibrate, ramp=rp.run_ramp, checks=rp.pre_run_checks, night="", say=print,
               model_path=PROD_MODEL, results=None, states=None, work=None):
    """Run the jobs in order (skipping those with a result row). Returns the exit code (0 / 2 / 3)."""
    results, states, work = results or RESULTS[set_], states or STATES[set_], work or WORK[set_]
    if stop_reached(set_, n_invalid(results)):
        say(f"STOP (V02_PLAN 8.12 item 8): {n_invalid(results)} invalid {set_} points already (> {MAX_INVALID[set_]}); not running")
        return 3
    done = {r["id"] for r in read_rows(results)}
    todo = [j for j in jobs if j["id"] not in done]
    t0 = time.time()
    ramp_say = (lambda m: None) if set_ in BLIND else say   # T: the per-state lines (tau, event) are not printed
    for i, job in enumerate(todo, 1):
        model = init_model(model_path)
        failed, _, _ = calibrate_pairs(job, model, calibrate, say, model_path)
        own = [r for r in read_rows(CALIB_CSV) if r["for_point"] == job["id"]]     # D-v02-12a: the point's own calibrations
        n_cal, cal_s = len(own), round(sum(float(r["seconds"]) for r in own), 1)
        row = dict(id=job["id"], set=set_, chi=job["chi"], syz=job["syz"], model_key=job["model_key"], provenance=job["provenance"],
                   source=f"run_production.py --set {set_}", night=night, n_new_calib=n_cal, calib_seconds=cal_s)
        if failed:
            row.update(valid="no", reason="gate_a_fail: " + " ".join(failed), ramp_seconds=0)
        else:
            efn, rn = entry_fn_for(job["syz"]), run_name(job)
            n, na, problems = checks([(job["chi"], job["syz"], rp.CELL_1X, rn, job["levels"])], model, entry_fn=efn, model_path=model_path)
            if problems:
                say(f"{rn}: pre-run checks found {len(problems)} problems, e.g. {problems[0]}; STOP (not scored)")
                return 2
            ts = time.time()
            summ = ramp(job["chi"], job["syz"], runner, model, states, work, ramp_say, rn=rn, levels=job["levels"], entry_fn=efn)
            ok, reasons = acceptance(summ)
            row.update(last_unmoved=summ["last_unmoved"], first_moved=summ["first_moved"], tau_P_mid=summ["mid"], event=summ["event"],
                       first_event=summ["first_event"], valid="yes" if ok else "no", reason=";".join(reasons),
                       below_top="yes" if summ["checks"]["below_top"] else "no", ramp_seconds=round(time.time() - ts, 1))
        row["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        append_row(results, row, mj.RESULT_FIELDS)
        el = time.time() - t0
        say(f"{set_} point {i}/{len(todo)} {job['id']} chi {job['chi']:+.3f} syz {job['syz']:+.5f} | "
            f"{'valid' if row['valid'] == 'yes' else 'INVALID ' + shown_reason(set_, row['reason'])} | {n_cal} new calibrations | elapsed {el:.0f} s | "
            f"est. left {el / i * (len(todo) - i):.0f} s")
        nb = n_invalid(results)
        if stop_reached(set_, nb):
            say(f"STOP (V02_PLAN 8.12 item 8): {nb} invalid {set_} points (> {MAX_INVALID[set_]}); stop for Ansh")
            return 3
    return 0


def placeholder_model(model, jobs):
    """Dry run only: a pair without an entry is rendered with the step-3 entry nearest in (chi / 30, sigma_yz / s), so the
    render check sees realistic coefficient signs and sizes. The real run renders with the real entries after calibration."""
    m = dict(correction=dict(model["correction"]))
    have = [(float(k.split(",")[0]), float(k.split(",")[1]), k) for k in model["correction"]]
    for j in jobs:
        for c, s in pairs_needed(j):
            k = r4.model_key(c, s)
            if k not in m["correction"]:
                near = min(have, key=lambda h: ((h[0] - c) / 30) ** 2 + ((h[1] - s) / mj.S) ** 2 + (0 if (h[1] == 0) == (s == 0) else 9))
                m["correction"][k] = model["correction"][near[2]]
    return m


def dry_run(set_, jobs, say=print):
    model = init_model() if os.path.exists(PROD_MODEL) else json.load(open(STEP3_MODEL))
    done = {r["id"] for r in read_rows(RESULTS[set_])}
    if set_ == "T":
        done |= {r["id"] for r in mj.step5_nodes(jobs)}           # imported at launch (or by tools/make_jobs_v02.py)
    todo = [j for j in jobs if j["id"] not in done]
    st = calib_status(model, read_rows(CALIB_CSV))
    new_pairs = sorted({r4.model_key(c, s) for j in todo for c, s in pairs_needed(j)} - set(st))
    pm = placeholder_model(model, todo)
    specs = [(j["chi"], j["syz"], rp.CELL_1X, run_name(j), j["levels"]) for j in todo]
    n, na, problems = 0, 0, []
    for j, sp in zip(todo, specs):
        a, b, p = rp.pre_run_checks([sp], pm, entry_fn=entry_fn_for(j["syz"]), model_path=PROD_MODEL)
        n, na, problems = n + a, na + b, problems + p
    say(f"{set_}: {len(jobs)} points in the set/batch, {len(done & {j['id'] for j in jobs})} done, {len(todo)} to run; "
        f"{len(new_pairs)} new calibrations needed; invalid so far {n_invalid(RESULTS[set_])} (stop if > {MAX_INVALID[set_]})")
    say(f"pre-run checks: {n} LAMMPS inputs rendered and parsed, {na} stability argvs parsed; problems: {len(problems)}"
        + (" (pairs without a calibration rendered with the nearest step-3 entry, dry run only)" if new_pairs else ""))
    for p in problems[:5]:
        say("   BAD " + p)
    est, mx = len(todo) * RAMP_S + len(new_pairs) * CALIB_S, len(todo) * RAMP_S_MAX + len(new_pairs) * CALIB_S_MAX
    say(f"estimate: {len(todo)} x {RAMP_S} s ramps + {len(new_pairs)} x {CALIB_S} s calibrations = {est / 3600:.1f} h "
        f"(at the step-5 / step-3 maxima {mx / 3600:.1f} h); 7 ranks, measured")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("T", "P"), required=True); ap.add_argument("--night", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if (a.set == "P") != (a.night is not None):
        ap.error("--night N is required with --set P and not allowed with --set T")
    tj, pj = mj.build()
    say = lambda m: print(m, flush=True)
    if a.set == "T":
        jobs = tj
        if not a.dry_run and mj.import_step5(tj):
            say("imported the 9 step-5 T nodes")
    else:
        done = {r["id"] for r in read_rows(RESULTS["P"])}
        ids = night_batch(a.night, pj, done, write=not a.dry_run)
        by_id = {j["id"]: j for j in pj}
        jobs = [by_id[i] for i in ids]
    if a.dry_run:
        return dry_run(a.set, jobs, say)
    init_model()
    runner = r4.LammpsRunner(os.environ.get("NP", "7"), **rp.CELL_1X, model_path=PROD_MODEL)
    code = run_points(a.set, jobs, runner, LammpsCalibrator(os.environ.get("NP", "7")), night=a.night or "", say=say)
    say(counts_summary(a.set, RESULTS[a.set]))
    say(f"PRODUCTION --set {a.set}{f' --night {a.night}' if a.night else ''}: exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
