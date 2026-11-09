"""Fake-data tests of scripts/v02_cost.py (D-v02-16, D-v02-18). No real project data is read."""
import csv, hashlib, json, os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v02_cost as vc

K0, K1, K2, K3 = "10,+0.00", "10,+0.09", "20,+0.28", "30,-0.47"      # K0: sigma_yz = 0 key


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def run(method, seed, ids, reached=True, n_total=None):
    picks = len(ids) if reached else n_total + 1
    return dict(method=method, seed=seed, picks=picks, reached=reached, picked=list(range(len(ids))), picked_ids=list(ids))


def fake(tmp, states, calib, runs, n_total, results=None, stop_resume=None):
    """states: {point: [(seconds, key), ...]}; calib: [(key, seconds, verdict, for_point)]; runs: run() dicts."""
    pts = sorted(states)
    if results is None:
        results = [dict(id=p, set="P", valid="yes", reason="", night="1", ramp_seconds=sum(s for s, _ in states[p]))
                   for p in pts]
    write_csv(tmp / "P_results.csv", results, ["id", "set", "valid", "reason", "night", "ramp_seconds"])
    write_csv(tmp / "P_states.csv",
              [dict(run=f"{p}_chi+1.0000_syz+0.09375", model_key=k, seconds=s) for p in pts for s, k in states[p]],
              ["run", "model_key", "seconds"])
    write_csv(tmp / "calibrations.csv", [dict(model_key=k, seconds=s, verdict=v, for_point=fp) for k, s, v, fp in calib],
              ["model_key", "seconds", "verdict", "for_point"])
    by = {}
    for r in runs:
        by.setdefault(r["method"], []).append(r)
    json.dump(dict(n_total=n_total, runs=by), open(tmp / "final.json", "w"))
    sr = None
    if stop_resume is not None:
        sr = tmp / "stop_resume.json"
        json.dump(stop_resume, open(sr, "w"))
    return [str(tmp / n) for n in ("P_results.csv", "P_states.csv", "calibrations.csv", "final.json")] + [sr and str(sr)]


HAND_STATES = {"A": [(60, K0), (40, K1)], "B": [(100, K1), (100, K2)], "C": [(300, K3)], "D": [(400, K2)]}
HAND_CALIB = [(K1, 10, "PASS", "P1"), (K2, 20, "PASS", "P2"), (K3, 30, "PASS", "P3"), (K0, 5, "PASS", "T01")]


def compute(paths, **kw):
    return vc.compute(*paths[:4], stop_resume_path=paths[4], pool_runs_path=None, **kw)


def row(rows, method, seed):
    return next(r for r in rows if r["method"] == method and r["seed"] == seed)


def test_hand_example(tmp_path):
    p = fake(tmp_path, HAND_STATES, HAND_CALIB, [run("al", 130, ["A", "B"]), run("random", 130, ["C", "D", "A"])], 4)
    rows, s = compute(p)
    al, rnd = row(rows, "al", 130), row(rows, "random", 130)
    assert (al["point_s"], al["calib_s"], al["total_s"], al["n_keys"]) == (300, 30, 330, 2)
    assert al["core_h"] == pytest.approx(0.641667, abs=1e-6)
    assert (rnd["point_s"], rnd["calib_s"], rnd["total_s"], rnd["n_keys"]) == (800, 60, 860, 3)
    assert rnd["core_h"] == pytest.approx(1.672222, abs=1e-6)
    b = s["baseline"]
    assert (b["point_s"], b["calib_s"], b["total_s"]) == (1000, 60, 1060)
    assert s["saving_a_vs_baseline"]["al"]["median_s"] == 730
    assert s["saving_a_vs_baseline"]["al"]["median_pct_of_baseline"] == 68.9
    sb = s["saving_b_al_vs_best_other"]
    assert sb["best_other"] == "random" and sb["median_s"] == 530 and sb["per_seed"][0]["saving_s"] == 530


def test_item2_duplicate_key_pass_row_and_t_row(tmp_path):
    calib = [(K1, 99, "FAIL", "P1"), (K1, 10, "PASS", "P1"), (K2, 20, "PASS", "T05")]
    p = fake(tmp_path, {"A": [(100, K1), (50, K2)]}, calib, [run("al", 130, ["A"])], 1)
    rows, _ = compute(p)
    assert row(rows, "al", 130)["calib_s"] == 30                       # PASS row of K1 (10) + T-made row of K2 (20)
    assert vc.is_zero_key(K0) and not vc.is_zero_key(K1)


def test_item4_never_reached_full_budget_lower_bound(tmp_path):
    runs = [run("al", 130, ["A"]), run("random", 130, ["A", "B", "C"], reached=False, n_total=3)]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(20, K1)], "C": [(30, K1)]}, [(K1, 1, "PASS", "P1")], runs, 3)
    rows, s = compute(p)
    r = row(rows, "random", 130)
    assert r["reached"] == "not reached (>=)" and r["n_points"] == 3 and r["total_s"] == 61 and r["picks"] == 4
    assert s["saving_b_al_vs_best_other"]["best_other"] is None       # random never reached: not a best-other candidate
    assert vc.bound_note(row(rows, "al", 130), r) == "lower bound (other not reached)"
    assert s["saving_a_vs_baseline"]["random"]["not_reached"] == 1


def test_item5_best_other_lowest_median_not_per_seed_minimum(tmp_path):
    states = {"P50": [(50, K1)], "P150": [(150, K1)], "P200": [(200, K1)], "P300": [(300, K1)], "P10": [(10, K1)],
              "P5": [(5, K1)]}
    runs = [run("al", s, ["P10"]) for s in (130, 131, 132)]
    runs += [run("random", 130, ["P200"]), run("random", 131, ["P200"]), run("random", 132, ["P50"])]          # median 200
    runs += [run("spacefill", 130, ["P150"]), run("spacefill", 131, ["P300"]), run("spacefill", 132, ["P300"])]  # median 300
    runs += [run("grid", 0, ["P5", "P10", "P50", "P150", "P200", "P300"], reached=False, n_total=6)]      # never reached
    p = fake(tmp_path, states, [(K1, 0, "PASS", "P1")], runs, 6)
    _, s = compute(p)
    sb = s["saving_b_al_vs_best_other"]
    assert sb["best_other"] == "random"                                  # per-seed minimum at 130 would be spacefill (150)
    assert sb["per_seed"][0]["other_s"] == 200 and sb["per_seed"][0]["saving_s"] == 190
    assert sb["median_s"] == 190 and sb["min_s"] == 40 and sb["max_s"] == 190


def test_item6_grid_substitute_costed_as_simulated(tmp_path):
    g = run("grid", 0, ["A", "S"])                                        # node B invalid, substituted by S
    p = fake(tmp_path, {"A": [(10, K1)], "S": [(70, K1)]}, [(K1, 0, "PASS", "P1")], [g, run("al", 130, ["A"])], 3)
    rows, _ = compute(p)
    assert row(rows, "grid", 0)["point_s"] == 80


def test_item7_grid_paired_with_every_seed(tmp_path):
    runs = [run("al", s, ["A"]) for s in (130, 131)] + [run("grid", 0, ["B"])]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(100, K1)]}, [(K1, 0, "PASS", "P1")], runs, 2)
    _, s = compute(p)
    sb = s["saving_b_al_vs_best_other"]
    assert sb["best_other"] == "grid" and [x["other_s"] for x in sb["per_seed"]] == [100, 100]


def test_item8_picked_count_mismatch_stops(tmp_path):
    bad = run("al", 130, ["A", "B"]); bad["picks"] = 3
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(10, K1)]}, [(K1, 0, "PASS", "P1")], [bad], 4)
    with pytest.raises(SystemExit, match="item 8"):
        compute(p)
    assert vc.check_pick_count(run("al", 130, ["A", "B"]), 4) == []


def test_item9_invalid_point_warning(tmp_path):
    res = [dict(id="A", set="P", valid="yes", reason="", night="1", ramp_seconds=10),
           dict(id="B", set="P", valid="no", reason="not_returned", night="1", ramp_seconds=10)]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(10, K1)]}, [(K1, 0, "PASS", "P1")], [run("al", 130, ["A"])], 2, results=res)
    _, s = compute(p)
    assert s["invalid"]["n_invalid"] == 1 and "understated by up to 1 points" in s["invalid"]["warning"]


def test_item10_gate_a_failure_baseline_only(tmp_path):
    res = [dict(id="A", set="P", valid="yes", reason="", night="1", ramp_seconds=10),
           dict(id="F", set="P", valid="no", reason="gate_a_fail: 30,+0.47", night="1", ramp_seconds=0)]
    calib = [(K1, 10, "PASS", "A"), ("30,+0.47", 25, "FAIL", "F")]
    p = fake(tmp_path, {"A": [(100, K1)]}, calib, [run("al", 130, ["A"])], 2, results=res)
    rows, s = compute(p)
    assert s["baseline"]["total_s"] == 135 and s["baseline"]["n_gate_a_failed"] == 1
    assert row(rows, "al", 130)["total_s"] == 110


def test_item11_lost_seconds_charged_to_no_method(tmp_path):
    args = ({"A": [(100, K1)]}, [(K1, 10, "PASS", "P1")], [run("al", 130, ["A"])], 1)
    rows0, s0 = compute(fake(tmp_path, *args))
    lost = [dict(night=4, point_id="A", lost_seconds=36)]
    rows1, s1 = compute(fake(tmp_path, *args, stop_resume=lost))
    assert rows1 == rows0 and s1["baseline"] == s0["baseline"]
    assert s1["lost_to_stop_resume"]["total_s"] == 36 and s1["lost_to_stop_resume"]["charged_to"] == "no method"
    assert s1["lost_to_stop_resume"]["core_h"] == pytest.approx(0.07)


def test_item13_baseline_includes_invalid_points(tmp_path):
    res = [dict(id="A", set="P", valid="yes", reason="", night="1", ramp_seconds=10),
           dict(id="B", set="P", valid="no", reason="not_returned", night="1", ramp_seconds=40)]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(40, K2)]}, [(K1, 1, "PASS", "A"), (K2, 2, "PASS", "B")],
             [run("al", 130, ["A"])], 2, results=res)
    _, s = compute(p)
    assert (s["baseline"]["point_s"], s["baseline"]["calib_s"], s["baseline"]["n_points"]) == (50, 3, 2)


def test_item14_outputs_and_input_hashes(tmp_path):
    p = fake(tmp_path, HAND_STATES, HAND_CALIB, [run("al", 130, ["A", "B"])], 4)
    out = tmp_path / "cost"
    vc.main(["--results", p[0], "--states", p[1], "--calib", p[2], "--final", p[3], "--stop-resume", str(tmp_path / "none"),
             "--pool-runs", str(tmp_path / "none"), "--out", str(out)])
    s = json.load(open(out / "cost_summary.json"))
    for name, path in zip(("P_results", "P_states", "calibrations", "final"), p[:4]):
        assert s["inputs"][name]["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert next(csv.reader(open(out / "cost_runs.csv"))) == vc.RUN_COLUMNS
    assert s["al_selection_compute"]["included"] is False


def test_item16_core_hours_and_price(tmp_path):
    assert vc.core_h(3600) == 7.0
    p = fake(tmp_path, HAND_STATES, HAND_CALIB, [run("al", 130, ["A", "B"])], 4)
    _, s = compute(p)
    assert "price" not in s and s["baseline"]["core_h"] == pytest.approx(1060 * 7 / 3600, abs=1e-6)
    _, s = compute(p, price=2.0, price_date="2026-11-06", price_source="https://example.org/price")
    assert s["price"]["baseline_rupees"] == pytest.approx(2.0 * s["baseline"]["core_h"])
    with pytest.raises(SystemExit, match="price-date"):
        compute(p, price=2.0)


def test_no_lammps_env_imports():
    src = open(os.path.join(ROOT, "scripts", "v02_cost.py")).read()
    for mod in ("run_step4", "run_production", "lammps", "ramp_v02", "calibrate_yz", "v02_compare"):
        assert f"import {mod}" not in src and f"from {mod}" not in src


def test_d19_al_not_reached_upper_bound(tmp_path):
    """D-v02-19 line 1: AL never reaches -> costed at its full budget (>=), saving (b) marked 'upper bound (AL not reached)'."""
    runs = [run("al", 130, ["A", "B"], reached=False, n_total=2), run("random", 130, ["B"])]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(100, K1)]}, [(K1, 5, "PASS", "P1")], runs, 2)
    rows, s = compute(p)
    al = row(rows, "al", 130)
    assert al["reached"] == "not reached (>=)" and al["n_points"] == 2 and al["total_s"] == 115
    sb = s["saving_b_al_vs_best_other"]
    assert sb["best_other"] == "random" and sb["per_seed"][0]["bound"] == "upper bound (AL not reached)"
    assert sb["per_seed"][0]["saving_s"] == -10


def test_d19_no_pass_row_uses_last_row(tmp_path):
    """D-v02-19 line 2: a key with no PASS row uses its last row."""
    assert vc.key_seconds([dict(model_key=K1, seconds="7", verdict="FAIL", for_point="F"),
                           dict(model_key=K1, seconds="9", verdict="FAIL", for_point="F")]) == {K1: 9.0}
    assert vc.key_seconds([dict(model_key=K1, seconds="7", verdict="PASS", for_point="A"),
                           dict(model_key=K1, seconds="9", verdict="FAIL", for_point="F")]) == {K1: 7.0}


def test_d19_al_estimate_single_thread_core_h(tmp_path):
    """D-v02-19 line 3: no AL timing in --final -> 'not included' + estimate = median AL seconds at pool 376, s / 3600 core-h."""
    pool = [dict(method="al", pool="376", seconds=s) for s in ("8.0", "9.0", "10.0")]
    pool += [dict(method="al", pool="360", seconds="100"), dict(method="random", pool="376", seconds="100")]
    json.dump(pool, open(tmp_path / "pool.json", "w"))
    final = dict(runs=dict(al=[run("al", 130, ["A"])]))
    out = vc.al_selection_compute(final, str(tmp_path / "pool.json"))
    assert out["included"] is False and out["note"].startswith("not included")
    e = out["estimate"]
    assert e["label"] == "estimate" and e["median_s_per_run"] == 9.0 and e["n_runs"] == 3 and e["threads"] == 1
    assert e["core_h_per_run"] == pytest.approx(9.0 / 3600, abs=1e-6) and e["core_h_per_run"] != pytest.approx(vc.core_h(9.0))


def test_d20_neither_reached_not_defined(tmp_path):
    """D-v02-20 line 1: neither AL nor the best other reached on a seed -> 'not defined', no number."""
    runs = [run("al", 130, ["A"]), run("random", 130, ["B"]),
            run("al", 131, ["A", "B"], reached=False, n_total=2), run("random", 131, ["A", "B"], reached=False, n_total=2)]
    p = fake(tmp_path, {"A": [(10, K1)], "B": [(100, K1)]}, [(K1, 0, "PASS", "P1")], runs, 2)
    _, s = compute(p)
    sb = s["saving_b_al_vs_best_other"]
    nd = sb["per_seed"][1]
    assert nd["seed"] == 131 and nd["bound"] == "not defined" and nd["saving_s"] is None and nd["saving_core_h"] is None
    assert sb["n_neither_reached_not_defined"] == 1 and sb["n_exact"] == 1 and sb["median_s"] == 90


def test_d20_median_exact_seeds_only(tmp_path):
    """D-v02-20 line 2: median and range over seeds where both reached; counts of the left-out seeds always shown."""
    st = {"A": [(10, K1)], "B": [(20, K1)], "C": [(1000, K1)], "D": [(2000, K1)], "E": [(3000, K1)]}
    runs = [run("al", 130, ["A"]), run("random", 130, ["C"]),                         # exact: 990
            run("al", 131, ["B"]), run("random", 131, ["C"]),                         # exact: 980
            run("al", 132, ["A"]), run("random", 132, ["C", "D", "E", "A", "B"], reached=False, n_total=5),   # lower: 6020
            run("al", 133, ["B"]), run("random", 133, ["C", "D", "E", "A", "B"], reached=False, n_total=5),   # lower: 6010
            run("al", 134, ["A", "B", "C", "D", "E"], reached=False, n_total=5), run("random", 134, ["D"])]  # upper: -4030
    p = fake(tmp_path, st, [(K1, 0, "PASS", "P1")], runs, 5)
    _, s = compute(p)
    sb = s["saving_b_al_vs_best_other"]
    assert sb["best_other"] == "random"
    assert sb["median_s"] == 985 and sb["min_s"] == 980 and sb["max_s"] == 990     # all five seeds would give median 990
    assert (sb["n_exact"], sb["n_al_not_reached_upper_bound"], sb["n_other_not_reached_lower_bound"],
            sb["n_neither_reached_not_defined"]) == (2, 1, 2, 0)
    assert [x["bound"] for x in sb["per_seed"]] == ["", "", "lower bound (other not reached)",
                                                    "lower bound (other not reached)", "upper bound (AL not reached)"]
