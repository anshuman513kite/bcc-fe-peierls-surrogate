"""Step 6 of V02_PLAN (2x cell check) driver, as defined in step-table row 6 (L78) and section 8.11 D-v02-6 (c, d), 3 Nov 2026.

Runs (2x cell, sims/03_peierls_chi0_2x/relaxed_down.data, glide length 293.75 A, nz 42): chi = 0 at sigma_yz = 0 (the D6 test
deferred from v0.1, corrected loading) and chi = 0 at sigma_yz = +s and -s; same ramp, D5 rule, stability seeds and step-3
entries as step 5 (ramp_v02.py). The carry of the perfect-slab calibration to the 2x cell is an ASSUMPTION (intensive
quantities; surface forces from the 2x cell's own area and slab); the test cannot separate a loading error from a size effect.
Pass (D-v02-6 c, D-v02-7 item 5), per 2x run: bracket midpoint within 3 % of its 1x value (step 5: phase A chi 0, corners
  chi 0 +-s, from checks/12_corner_gate/step5_summary_s<s>.json); the same first-event D5 class as the 1x run; both stability
  seeds return. The absolute difference is reported against 0.02 GPa (descriptive).
  Failure -> STOP for Ansh (exit 2). Chaining (D-v02-6 d): runs only if step 5 is PASS at s = 0.75 (checked here too).
Usage (from checks/13_size_gate): python run_step6.py [--dry-run]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../12_corner_gate"))
import ramp_v02 as rp
r4 = rp.r4
S, TOL_REL, TOL_ABS = 0.75, 0.03, 0.02
WORK = os.path.join(rp.ROOT, "sims/11_size_gate")
RESULTS = os.path.join(HERE, "step6_results.csv")
STEP5 = os.path.join(rp.ROOT, "checks/12_corner_gate/step5_summary_s0.75.json")
RUNS = [(0, 0.0), (0, +S), (0, -S)]


def one_x(step5):
    """1x run summaries (step 5) for the three step-6 runs."""
    return {(0, 0.0): step5["phaseA"]["0"], (0, +S): step5["corners"][f"0,{+S:+.2f}"], (0, -S): step5["corners"][f"0,{-S:+.2f}"]}


def verdict(two, one):
    """two, one: {(chi, syz): run summary (ramp_v02.summarise)}. Returns (verdict, exit, rows)."""
    rows, ok = [], True
    for k in RUNS:
        t, o = two.get(k) or {}, one[k]
        m2, m1 = t.get("mid"), o["mid"]
        rel = None if m2 is None else abs(m2 - m1) / m1
        same_event = t.get("first_event") is not None and t.get("first_event") == o.get("first_event")
        seeds = bool(t.get("checks", {}).get("stability"))
        good = rel is not None and rel < TOL_REL and same_event and seeds       # D-v02-6 c, D-v02-7 item 5
        ok &= good
        rows.append(dict(run=k, mid_2x=m2, mid_1x=m1, rel=rel, abs=None if m2 is None else abs(m2 - m1), event_2x=t.get("first_event"),
                         event_1x=o.get("first_event"), same_event=same_event, seeds_returned=seeds, pass_=good,
                         abs_vs_0p02="(no bracket)" if m2 is None else ("<= 0.02" if abs(m2 - m1) <= TOL_ABS + 1e-9 else "> 0.02")))
    return ("PASS", 0, rows) if ok else ("STOP", 2, rows)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    model = json.load(open(r4.MODEL))
    specs = [(c, s, rp.CELL_2X) for c, s in RUNS]
    n, na, problems = rp.pre_run_checks(specs, model)
    print(f"pre-run checks (2x cell): {n} LAMMPS inputs rendered and parsed, {na} stability argvs parsed; problems: {len(problems)}", flush=True)
    for p in problems[:10]:
        print("   BAD " + p, flush=True)
    if a.dry_run:
        for c, s, _ in specs:
            print(f"   run {rp.run_name(c, s)} (2x): max {rp.max_states(s)} states", flush=True)
        return 1 if problems else 0
    if problems:
        print("pre-run checks FAILED; not running", flush=True); return 2
    if not os.path.exists(STEP5) or json.load(open(STEP5))["verdict"] != "PASS":
        print("step 5 is not PASS at s = 0.75: step 6 not run (D-v02-6 d)", flush=True); return 4
    runner = r4.LammpsRunner(os.environ.get("NP", "7"), **rp.CELL_2X)
    two = {(c, s): rp.run_ramp(c, s, runner, model, RESULTS, WORK, lambda m: print(m, flush=True)) for c, s in RUNS}
    v, code, rows = verdict(two, one_x(json.load(open(STEP5))))
    for r in rows:
        print(f"2x {r['run']}: mid {r['mid_2x']} vs 1x {r['mid_1x']}: rel {r['rel']}, abs {r['abs']} ({r['abs_vs_0p02']}); "
              f"event 2x {r['event_2x']} vs 1x {r['event_1x']}; seeds {r['seeds_returned']}; {'pass' if r['pass_'] else 'FAIL'}", flush=True)
    print(f"STEP 6 (V02_PLAN 8.11): {v}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
