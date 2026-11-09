"""Step 5 of V02_PLAN (range probes) driver, as defined in section 5.4 rules 3-4 and section 8.11 D-v02-6 (3 Nov 2026).

Phase A (D-v02-6 a), FIRST: sigma_yz = 0 at chi -30, 0, +30 (146.87 A cell, step-3 entries "<chi>,+0.00"). Each bracket
  midpoint must be within 0.02 GPa of v0.1 (-30: 1.160-1.170, 0: 1.310-1.320 B2, +30: 2.040-2.050 GPa) AND each run must pass
  rule 3 (first event a D5 glide jump or easy -> split, no anomaly, both seeds return; D-v02-7 item 1); otherwise STOP before
  the corners (exit 2). These are tau_P(0) for rule 4.
Phase B: the six corners chi {-30, 0, +30} x sigma_yz {+s, -s} (s = 0.75 unless --s), step-3 entries "<chi>,<+-s>".
  Each corner (rule 3, ramp_v02.summarise): bracket found, first-moved tau <= 2.95 GPa, first event a glide jump or
  easy -> split (no anomaly), no move during the sigma_yz pre-ramp, both stability seeds return.
Verdict:
  all six pass and NOT rule 4 -> PASS at s (exit 0);
  rule 4 (|tau_P(+-s) - tau_P(0)| < 0.02 GPa at all three angles, both signs) -> STOP for Ansh (exit 3);
  any corner fails, s > 0.25 -> INCOMPLETE: all six repeat at s - 0.25 (run_step5.py --s <s - 0.25>) plus the step-4
    full-sequence follow-up at that level (D-v02-5 b) (exit 1);
  any corner fails at s = 0.25, or phase A fails -> STOP for Ansh (exit 2).
Checkpoint step5_results[_s<s>].csv (one line per state), resume, render/argv pre-run checks: see ramp_v02.py.
Usage (from checks/12_corner_gate; launch with tools/launch_safe.sh, NP from the launcher):
  python run_step5.py --dry-run [--s 0.75]
  python run_step5.py [--s 0.75]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ramp_v02 as rp
r4 = rp.r4

CHIS, REF = (-30, 0, 30), {-30: (1.160, 1.170), 0: (1.310, 1.320), 30: (2.040, 2.050)}
TOL_REPRO, TOL_RULE4 = 0.02, 0.02
WORK = os.path.join(rp.ROOT, "sims/10_corner_gate")


def results_path(s):
    return os.path.join(HERE, "step5_results.csv" if abs(s - 0.75) < 1e-9 else f"step5_results_s{s:.2f}.csv")


def specs(s):
    a = [(c, 0.0, rp.CELL_1X) for c in CHIS]
    b = [(c, sg * s, rp.CELL_1X) for c in CHIS for sg in (+1, -1)]
    return a, b


def verdict(a_sum, b_sum, s):
    """a_sum: {chi: summary} for sigma_yz = 0; b_sum: {(chi, syz): summary} for the corners. Returns (verdict, exit, note)."""
    bad_a = [f"chi {c}: mid {a_sum[c]['mid']} vs v0.1 {sum(REF[c]) / 2:.3f}" for c in CHIS
             if a_sum[c]["mid"] is None or abs(a_sum[c]["mid"] - sum(REF[c]) / 2) > TOL_REPRO + 1e-9]
    if bad_a:
        return "STOP", 2, "phase A (sigma_yz = 0 reproduction) differs from v0.1 by > 0.02 GPa: " + "; ".join(bad_a)
    bad_r3 = [f"chi {c}: {[k for k, ok in a_sum[c].get('checks', {}).items() if not ok] or 'rule 3'}" for c in CHIS
              if not a_sum[c]["rule3_pass"]]
    if bad_r3:                                          # D-v02-7 item 1: phase A also requires rule 3
        return "STOP", 2, "phase A fails rule 3: " + "; ".join(bad_r3)
    if b_sum is None:
        return "INCOMPLETE", 1, "corners not run"
    failed = [k for k, v in b_sum.items() if not v["rule3_pass"]]
    if failed:
        if s - 0.25 < 0.25 - 1e-9:
            return "STOP", 2, f"corner(s) failed at s = {s}: {failed}; no lower level (stop for Ansh)"
        return "INCOMPLETE", 1, (f"corner(s) failed at s = {s}: {failed}; repeat ALL six at s = {s - 0.25:.2f} "
                                 f"(run_step5.py --s {s - 0.25:.2f}) plus the step-4 full-sequence follow-up at {s - 0.25:.2f} (D-v02-5 b)")
    d = {k: abs(v["mid"] - a_sum[k[0]]["mid"]) for k, v in b_sum.items()}
    if all(x < TOL_RULE4 for x in d.values()):
        return "STOP", 3, f"rule 4: |tau_P(+-s) - tau_P(0)| < 0.02 GPa at all three angles ({ {str(k): round(x, 3) for k, x in d.items()} })"
    return "PASS", 0, f"s = {s}"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--s", type=float, default=0.75)
    a = ap.parse_args(argv)
    model = json.load(open(r4.MODEL))
    A, B = specs(a.s)
    n, na, problems = rp.pre_run_checks(A + B, model)
    print(f"pre-run checks: {n} LAMMPS inputs rendered and parsed, {na} stability argvs parsed; problems: {len(problems)}", flush=True)
    for p in problems[:10]:
        print("   BAD " + p, flush=True)
    if a.dry_run:
        for c, s, _ in A + B:
            print(f"   run {rp.run_name(c, s)}: max {rp.max_states(s)} states (start, {len(rp.preyz_levels(s))} pre-ramp, up to "
                  f"{len(rp.COARSE_TAUS)} coarse + 4 fine, 2 stability)", flush=True)
        return 1 if problems else 0
    if problems:
        print("pre-run checks FAILED; not running", flush=True); return 2
    res, say = results_path(a.s), (lambda m: print(m, flush=True))
    runner = r4.LammpsRunner(os.environ.get("NP", "7"), **rp.CELL_1X)
    a_sum = {c: rp.run_ramp(c, s, runner, model, res, WORK, say) for c, s, _ in A}
    v, code, note = verdict(a_sum, None, a.s)
    if v == "STOP":
        print(f"STEP 5 (V02_PLAN 8.11): {v} — {note}", flush=True); return code
    b_sum = {(c, s): rp.run_ramp(c, s, runner, model, res, WORK, say) for c, s, _ in B}
    print("\nrun              bracket (GPa)      mid    event        first event  checks", flush=True)
    for k, x in list({(c, 0.0): v_ for c, v_ in a_sum.items()}.items()) + list(b_sum.items()):
        print(f"{x['run']:16s} {x['last_unmoved']}-{x['first_moved']}  {x['mid']}  {x['event']}  {x['first_event']}  "
              f"{'all ok' if x['rule3_pass'] else [c for c, ok in x['checks'].items() if not ok]}", flush=True)
    v, code, note = verdict(a_sum, b_sum, a.s)
    json.dump(dict(s=a.s, phaseA={str(c): x for c, x in a_sum.items()}, corners={f"{c},{s:+.2f}": x for (c, s), x in b_sum.items()},
                   verdict=v, note=note), open(os.path.join(HERE, f"step5_summary_s{a.s:.2f}.json"), "w"), indent=1)
    print(f"STEP 5 (V02_PLAN 8.11): {v} — {note}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
