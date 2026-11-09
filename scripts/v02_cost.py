"""v0.2 cost meter (V02_PLAN 8.12, D-v02-16 and D-v02-18): descriptive cost of each method's run to its first crossing.

Descriptive only: the Q8 verdict stays the frozen rule on pick counts (D-v02-16 (4)). Reads the --final output of
scripts/v02_compare.py (data/v02_compare/final.json) plus P_results.csv, P_states.csv and calibrations.csv; writes
cost_runs.csv and cost_summary.json (D-v02-18 item 14). Pure Python + the standard library: no LAMMPS-env module is
imported, so it runs and is tested in the ml env.

Usage (ml env): python scripts/v02_cost.py [--final data/v02_compare/final.json] [--out data/production/cost]
                [--price RUPEES_PER_CORE_H --price-date YYYY-MM-DD --price-source URL]
"""
import argparse, csv, hashlib, json, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NP = 7                                                                # MPI ranks of every production run (D-v02-16 (2))
OTHERS = ("random", "spacefill", "grid")                              # D-v02-18 item 5
NOT_REACHED = "not reached (>=)"                                      # D-v02-18 item 4
RUN_COLUMNS = ["method", "seed", "picks", "reached", "n_points", "point_s", "n_keys", "calib_s", "total_s", "core_h"]
DEFAULTS = dict(
    results=os.path.join(ROOT, "data", "production", "P_results.csv"),
    states=os.path.join(ROOT, "data", "production", "P_states.csv"),
    calib=os.path.join(ROOT, "checks", "14_production", "calibrations.csv"),
    final=os.path.join(ROOT, "data", "v02_compare", "final.json"),
    stop_resume=os.path.join(ROOT, "data", "production", "stop_resume.json"),
    pool_runs=os.path.join(ROOT, "data", "synthetic_v02", "pool_study_runs.json"),
    out=os.path.join(ROOT, "data", "production", "cost"),
)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- item 16: units
def core_h(seconds, np_=NP):
    """Item 16: core-hours = seconds x 7 / 3600."""
    return seconds * np_ / 3600.0


def money(core_hours, price):
    """Item 16: rupees = core-hours x price per core-hour (only when a price is given)."""
    return None if price is None else core_hours * price


# ---------------------------------------------------------------- item 1: point seconds
def point_id(run_name):
    """P_states.csv 'run' is '<ID>_chi..._syz...'; the point ID is the part before the first '_'."""
    return run_name.split("_")[0]


def point_seconds(states):
    """Item 1: a point's cost = the sum of its P_states 'seconds' over all its states (all phases)."""
    out = {}
    for r in states:
        p = point_id(r["run"])
        out[p] = out.get(p, 0.0) + float(r["seconds"])
    return out


# ---------------------------------------------------------------- item 2: calibration keys
def is_zero_key(key):
    """Item 2: a sigma_yz = 0 key ('chi,+0.00') uses the exact zero correction and needs no calibration."""
    return float(key.split(",")[1]) == 0.0


def point_keys(states):
    """Item 2: the distinct non-zero model_keys used by each point's states."""
    out = {}
    for r in states:
        s = out.setdefault(point_id(r["run"]), set())
        if not is_zero_key(r["model_key"]):
            s.add(r["model_key"])
    return out


def key_seconds(calib_rows):
    """Item 2: each key's measured seconds from calibrations.csv, whichever point (P or T) the row was made for. A key with
    more than one row: its PASS row is used (the last PASS row if several); a key without a PASS row: its last row."""
    rows = {}
    for r in calib_rows:
        rows.setdefault(r["model_key"], []).append(r)
    out = {}
    for k, rs in rows.items():
        passed = [r for r in rs if r["verdict"] == "PASS"]
        out[k] = float((passed or rs)[-1]["seconds"])
    return out


def run_keys(point_ids, pkeys):
    """Item 2: the distinct non-zero keys over the given points, each counted once."""
    ks = set()
    for p in point_ids:
        ks |= pkeys.get(p, set())
    return sorted(ks)


# ---------------------------------------------------------------- items 3, 4, 6, 8: the picked points of a run
def costed_points(run):
    """Items 3, 4, 6: the points a run is costed at = its stored picked_ids: up to and including the first crossing, or all
    its picks if it never crossed (full budget); for grid these already hold any substitute (the point actually simulated)."""
    return list(run["picked_ids"])


def reached_label(run):
    """Item 4: 'yes', or 'not reached (>=)' (the cost is a lower bound on the cost to the target)."""
    return "yes" if run["reached"] else NOT_REACHED


def check_pick_count(run, n_total):
    """Item 8: every training point is counted, starting design included: the stored picked list must hold exactly the
    run's pick count (starting points + added picks), with no repeats; for a run that never crossed (picks = n_total + 1)
    at most n_total points. Returns a list of problems (empty = consistent)."""
    tag = f"{run['method']} seed {run['seed']}"
    pk, ids = list(run["picked"]), list(run["picked_ids"])
    bad = []
    if len(ids) != len(pk):
        bad.append(f"{tag}: {len(pk)} picked indices but {len(ids)} picked IDs")
    if len(set(ids)) != len(ids):
        bad.append(f"{tag}: repeated picked IDs")
    if run["reached"]:
        if len(ids) != run["picks"]:
            bad.append(f"{tag}: picks {run['picks']} but {len(ids)} picked points")
    elif run["picks"] != n_total + 1 or len(ids) > n_total:
        bad.append(f"{tag}: not reached but picks {run['picks']} / {len(ids)} picked points (n_total {n_total})")
    return bad


def run_cost(run, pt_s, pkeys, kse):
    """Items 1-4, 16: one row of cost_runs.csv."""
    pts = costed_points(run)
    missing = [p for p in pts if p not in pt_s]
    if missing:
        raise SystemExit(f"{run['method']} seed {run['seed']}: no P_states rows for {missing[:5]}")
    keys = run_keys(pts, pkeys)
    nokey = [k for k in keys if k not in kse]
    if nokey:
        raise SystemExit(f"{run['method']} seed {run['seed']}: no calibrations.csv row for keys {nokey[:5]}")
    ps_ = sum(pt_s[p] for p in pts)
    cs = sum(kse[k] for k in keys)
    return dict(method=run["method"], seed=run["seed"], picks=run["picks"], reached=reached_label(run), n_points=len(pts),
                point_s=round(ps_, 3), n_keys=len(keys), calib_s=round(cs, 3), total_s=round(ps_ + cs, 3),
                core_h=round(core_h(ps_ + cs), 6))


# ---------------------------------------------------------------- items 10, 13: simulate-all baseline
def gate_a_failed(results):
    """Item 10: P points that failed Gate A (reason 'gate_a_fail: ...'); they have no states, only calibrations."""
    return sorted(r["id"] for r in results if r["reason"].startswith("gate_a_fail"))


def baseline(results, pt_s, pkeys, calib_rows, kse):
    """Item 13 (and 10): all P points' state seconds, invalid points included, plus each distinct non-zero key once; a
    Gate A failed point adds only its calibration keys (rows made for it), and only here."""
    ids = [r["id"] for r in results if r["set"] == "P"]
    failed = set(gate_a_failed(results))
    keys = set(run_keys(ids, pkeys))
    keys |= {r["model_key"] for r in calib_rows if r["for_point"] in failed and not is_zero_key(r["model_key"])}
    ps_ = sum(pt_s.get(p, 0.0) for p in ids)
    cs = sum(kse[k] for k in keys)
    return dict(n_points=len(ids), n_gate_a_failed=len(failed), point_s=round(ps_, 3), n_keys=len(keys),
                calib_s=round(cs, 3), total_s=round(ps_ + cs, 3), core_h=round(core_h(ps_ + cs), 6))


# ---------------------------------------------------------------- saving (a): each method against the baseline
def saving_vs_baseline(rows, base_total):
    """D-v02-16 (3a): per method, baseline - run cost over its runs (median, min, max; % of the baseline). A run that never
    reached the target gets no saving (D-v02-16 (3)): it is counted, not included."""
    out = {}
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        s = [base_total - r["total_s"] for r in rs if r["reached"] == "yes"]
        out[m] = dict(runs=len(rs), not_reached=len(rs) - len(s))
        if s:
            md = statistics.median(s)
            out[m].update(median_s=round(md, 3), min_s=round(min(s), 3), max_s=round(max(s), 3),
                          median_core_h=round(core_h(md), 6), median_pct_of_baseline=round(100.0 * md / base_total, 1))
    return out


# ---------------------------------------------------------------- items 5, 7: AL against the best other method
def best_other(rows):
    """Item 5: the single method among random, spacefill and grid with the lowest median cost over the seeds, among its runs
    that reached the target (not the cheapest method per seed). Ties: first in OTHERS order. (None, None) if none reached."""
    best = (None, None)
    for m in OTHERS:
        c = [r["total_s"] for r in rows if r["method"] == m and r["reached"] == "yes"]
        if c and (best[1] is None or statistics.median(c) < best[1]):
            best = (m, statistics.median(c))
    return best


def other_row(rows, method, seed):
    """Item 7: grid is one deterministic run, so its single row is paired with every seed; others: the row of that seed."""
    rs = [r for r in rows if r["method"] == method and (method == "grid" or r["seed"] == seed)]
    if len(rs) != 1:
        raise SystemExit(f"{method}: expected one run for seed {seed}, found {len(rs)}")
    return rs[0]


def bound_note(al, other):
    """Item 4: a saving against a run that never reached is a lower bound; if AL itself never reached, an upper bound;
    D-v02-20 1: if neither reached, 'not defined'."""
    if other["reached"] != "yes" and al["reached"] != "yes":
        return "not defined"
    notes = []
    if other["reached"] != "yes":
        notes.append("lower bound (other not reached)")
    if al["reached"] != "yes":
        notes.append("upper bound (AL not reached)")
    return "; ".join(notes)


def paired_saving(rows):
    """Items 5, 7 (D-v02-16 (3b)): AL against the best other method, paired per seed; median and range; negatives kept.
    D-v02-20: a seed where neither reached gets no number ('not defined'); the median and range use only seeds where both
    reached, shown next to the counts of seeds left out (AL not reached, other not reached, neither); every seed is listed."""
    m, med = best_other(rows)
    if m is None:
        return dict(best_other=None, note="no other method reached the target in any run")
    per = []
    for al in sorted((r for r in rows if r["method"] == "al"), key=lambda r: r["seed"]):
        o = other_row(rows, m, al["seed"])
        b = bound_note(al, o)
        s = None if b == "not defined" else o["total_s"] - al["total_s"]
        per.append(dict(seed=al["seed"], al_s=al["total_s"], other_s=o["total_s"], saving_s=None if s is None else round(s, 3),
                        saving_core_h=None if s is None else round(core_h(s), 6), bound=b))
    sv = [p["saving_s"] for p in per if not p["bound"]]
    out = dict(best_other=m, best_other_median_s=round(med, 3), per_seed=per, n_exact=len(sv),
               n_al_not_reached_upper_bound=sum(p["bound"] == "upper bound (AL not reached)" for p in per),
               n_other_not_reached_lower_bound=sum(p["bound"] == "lower bound (other not reached)" for p in per),
               n_neither_reached_not_defined=sum(p["bound"] == "not defined" for p in per))
    if sv:
        out.update(median_s=round(statistics.median(sv), 3), min_s=min(sv), max_s=max(sv),
                   median_core_h=round(core_h(statistics.median(sv)), 6), min_core_h=round(core_h(min(sv)), 6),
                   max_core_h=round(core_h(max(sv)), 6), n_negative=sum(s < 0 for s in sv),
                   n_bounded=sum(bool(p["bound"]) for p in per))
    return out


# ---------------------------------------------------------------- items 9, 11, 12, 14, 15
def invalid_warning(results):
    """Item 9: invalid P points are skipped by the frozen v02_compare and not stored, so their cost is not in any run."""
    n = sum(1 for r in results if r["set"] == "P" and r["valid"] != "yes")
    if n == 0:
        return dict(n_invalid=0, warning=None)
    return dict(n_invalid=n, warning=f"cost may be understated by up to {n} points: the frozen v02_compare does not store "
                                     "skipped invalid points")


def lost_to_stop_resume(path):
    """Item 11: seconds lost to a killed in-flight state (from log times), reported separately and charged to no method."""
    entries = json.load(open(path)) if path and os.path.exists(path) else []
    tot = sum(float(e["lost_seconds"]) for e in entries)
    return dict(entries=entries, total_s=round(tot, 3), core_h=round(core_h(tot), 6), charged_to="no method")


def timing_label(results):
    """Item 12: seconds as recorded (night and daytime runs), labelled 'real load', not normalised."""
    by = {}
    for r in results:
        if r["set"] == "P" and r["valid"] == "yes":
            by.setdefault(r["night"], []).append(float(r["ramp_seconds"]))
    return dict(label="real load", normalised=False,
                ramp_mean_s_by_night={n: round(statistics.mean(v), 1) for n, v in sorted(by.items())})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_hashes(paths):
    """Item 14: the sha256 of every input file."""
    return {name: dict(path=os.path.relpath(p, ROOT) if p.startswith(ROOT) else p, sha256=sha256_file(p))
            for name, p in paths.items()}


def al_selection_compute(final, pool_runs_path, pool=376):
    """Item 15: the AL model's own compute, reported separately. --final records no per-run timing; if a later record has
    'seconds' they are used, else 'not included' plus an estimate (median wall seconds of the synthetic pool-study AL runs
    at the same pool size; single-threaded, so core-h = s / 3600), labelled 'estimate'."""
    al = final["runs"].get("al", [])
    if al and all("seconds" in r for r in al):
        s = [float(r["seconds"]) for r in al]
        return dict(included=True, source="--final run records", median_s=statistics.median(s), total_s=sum(s))
    out = dict(included=False, note="not included: --final records no per-run timing")
    if pool_runs_path and os.path.exists(pool_runs_path):
        s = [float(r["seconds"]) for r in json.load(open(pool_runs_path)) if r["method"] == "al" and int(r["pool"]) == pool]
        if s:
            md = statistics.median(s)
            out["estimate"] = dict(label="estimate", median_s_per_run=round(md, 1), core_h_per_run=round(md / 3600.0, 6),
                                   threads=1, n_runs=len(s), pool=pool,
                                   source=os.path.relpath(pool_runs_path, ROOT) + " (synthetic AL runs, stop at crossing)")
    return out


# ---------------------------------------------------------------- outputs and main
def write_outputs(out_dir, rows, summary):
    """Item 14: cost_runs.csv and cost_summary.json in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cost_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(os.path.join(out_dir, "cost_summary.json"), "w"), indent=1)


def compute(results_path, states_path, calib_path, final_path, stop_resume_path=None, pool_runs_path=None,
            price=None, price_date=None, price_source=None):
    """All of D-v02-18 on the given files; returns (cost_runs rows, summary)."""
    if price is not None and not (price_date and price_source):
        raise SystemExit("--price needs --price-date and --price-source (item 16)")
    results, states, calib = read_csv(results_path), read_csv(states_path), read_csv(calib_path)
    final = json.load(open(final_path))
    pt_s, pkeys, kse = point_seconds(states), point_keys(states), key_seconds(calib)
    runs = [r for m in final["runs"] for r in final["runs"][m]]
    bad = [b for r in runs for b in check_pick_count(r, final["n_total"])]
    if bad:
        raise SystemExit("picked-count check failed (item 8):\n  - " + "\n  - ".join(bad))
    rows = [run_cost(r, pt_s, pkeys, kse) for r in runs]
    base = baseline(results, pt_s, pkeys, calib, kse)
    sa, sb = saving_vs_baseline(rows, base["total_s"]), paired_saving(rows)
    paths = dict(P_results=results_path, P_states=states_path, calibrations=calib_path, final=final_path)
    if stop_resume_path and os.path.exists(stop_resume_path):
        paths["stop_resume"] = stop_resume_path
    summary = dict(
        note="descriptive only (D-v02-16 (4), D-v02-18): the Q8 verdict is the frozen rule on pick counts",
        rules="V02_PLAN 8.12 D-v02-16, D-v02-18", np=NP, core_h_formula="seconds x 7 / 3600",
        baseline=base, saving_a_vs_baseline=sa, saving_b_al_vs_best_other=sb,
        invalid=invalid_warning(results), lost_to_stop_resume=lost_to_stop_resume(stop_resume_path),
        timing=timing_label(results), al_selection_compute=al_selection_compute(final, pool_runs_path),
        inputs=input_hashes(paths))
    if price is not None:
        summary["price"] = dict(rupees_per_core_h=price, date=price_date, source=price_source,
                                baseline_rupees=money(base["core_h"], price),
                                saving_b_median_rupees=money(sb.get("median_core_h"), price) if "median_core_h" in sb else None)
    return rows, summary


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--results", default=DEFAULTS["results"])
    a.add_argument("--states", default=DEFAULTS["states"])
    a.add_argument("--calib", default=DEFAULTS["calib"])
    a.add_argument("--final", default=DEFAULTS["final"])
    a.add_argument("--stop-resume", default=DEFAULTS["stop_resume"])
    a.add_argument("--pool-runs", default=DEFAULTS["pool_runs"])
    a.add_argument("--out", default=DEFAULTS["out"])
    a.add_argument("--price", type=float, default=None, help="rupees per core-hour")
    a.add_argument("--price-date", default=None)
    a.add_argument("--price-source", default=None)
    a = a.parse_args(argv)
    rows, summary = compute(a.results, a.states, a.calib, a.final, a.stop_resume, a.pool_runs,
                            a.price, a.price_date, a.price_source)
    write_outputs(a.out, rows, summary)
    print(f"written {len(rows)} runs to {os.path.join(a.out, 'cost_runs.csv')} and cost_summary.json")
    return rows, summary


if __name__ == "__main__":
    main()
