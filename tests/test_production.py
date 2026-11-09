"""Night T / night P tests (V02_PLAN section 8.12, D-v02-8; 4 Nov 2026). No LAMMPS process is started: the driver runs with a
fake calibration, a fake ramp and a fake pre-run check."""
import csv, glob, json, os, re, sys
import numpy as np
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in ("tools", "src", "checks/10_gateA_yz", "checks/11_core_gate", "checks/12_corner_gate", "checks/14_production"):
    sys.path.insert(0, os.path.join(ROOT, p))
import make_jobs_v02 as mj
import run_production as rpd
import ramp_v02 as rp
import run_step4 as r4

TJ, PJ = mj.build()


def test_T_and_P_disjoint():
    assert len(TJ) == 49 and len(PJ) == 376
    t = {(round(j["chi"], 9), j["syz"]) for j in TJ}
    p = {(round(j["chi"], 9), j["syz"]) for j in PJ}
    assert not t & p and len(p) == 376
    assert not {j["syz"] for j in TJ} & {j["syz"] for j in PJ}
    assert sorted({j["syz"] for j in TJ}) == [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    assert sorted({j["syz"] for j in PJ}) == [-0.65625, -0.46875, -0.28125, -0.09375, 0.09375, 0.28125, 0.46875, 0.65625]
    assert {j["chi"] for j in PJ} >= {-30.0, 30.0}


def test_P_permutation_reproducible_from_seed():
    assert mj.PERM_SEED == 20261104
    perm = np.random.default_rng(20261104).permutation(376)
    assert [j["pool_index"] for j in PJ] == [int(k) for k in perm]
    _, pj2 = mj.build()
    assert [j["id"] for j in pj2] == [j["id"] for j in PJ]


def test_nine_T_nodes_imported(tmp_path):
    sealed = str(tmp_path / "sealed/T_results.csv")
    assert mj.import_step5(TJ, sealed) == 9
    assert mj.import_step5(TJ, sealed) == 0                           # idempotent
    rows = list(csv.DictReader(open(sealed)))
    assert len(rows) == 9 and sorted(r["provenance"] for r in rows).count("step5 corner") == 6
    d = json.load(open(mj.STEP5_SUMMARY))
    by = {(float(r["chi"]), float(r["syz"])): r for r in rows}
    for x in list(d["phaseA"].values()) + list(d["corners"].values()):
        r = by[(float(x["chi"]), float(x["syz"]))]
        assert float(r["tau_P_mid"]) == x["mid"] and r["valid"] == "yes" and "b1bec26" in r["source"]
    assert {r["id"] for r in rows} == {j["id"] for j in TJ if abs(j["chi"]) in (0, 30) and j["syz"] in (0.0, 0.75, -0.75)}


def test_night_cap_50_and_batch_fixed(tmp_path):
    b1 = rpd.night_batch(1, PJ, set(), prod=str(tmp_path))
    assert b1 == [j["id"] for j in PJ[:50]]
    assert rpd.night_batch(1, PJ, set(b1[:10]), prod=str(tmp_path)) == b1        # resumed night: same batch
    b2 = rpd.night_batch(2, PJ, set(b1), prod=str(tmp_path))
    assert b2 == [j["id"] for j in PJ[50:100]]
    last = rpd.night_batch(8, PJ, {j["id"] for j in PJ[:350]}, prod=str(tmp_path), write=False)
    assert len(last) == 26


def test_preramp_levels():
    for j in TJ:                                                       # T: identical to the steps 4-5 ladder
        assert j["levels"] == rp.preyz_levels(j["syz"])
    j = next(j for j in PJ if j["syz"] == -0.65625)
    assert j["levels"] == [-0.09375, -0.28125, -0.46875, -0.65625]
    for j in PJ:
        assert all(r4.model_key(j["chi"], lv) in {q["model_key"] for q in PJ} for lv in j["levels"])


def test_entry_fn_zero_only_at_start():
    model = {"correction": {"0,+0.25": {"q0": [1.0] * 5}}}
    f = rpd.entry_fn_for(0.09375)
    assert f(model, -29.333333333333332, 0.0) == rpd.ZERO_ENTRY
    with pytest.raises(KeyError):
        f(model, -29.333333333333332, 0.09375)
    with pytest.raises(KeyError):
        rpd.entry_fn_for(0.0)(model, 10.0, 0.0)                         # a sigma_yz = 0 target needs its real entry
    from loading_v02 import model_correction
    assert np.allclose(model_correction(rpd.ZERO_ENTRY, 0.0), 0.0)


def _summ(bracket=True, pre=True, first="glide_jump", stab=True, below=True):
    return dict(run="x", last_unmoved=1.0 if bracket else None, first_moved=1.01 if bracket else None, mid=1.005 if bracket else None,
                event=first if bracket else None, first_event=first,
                checks=dict(bracket=bracket, below_top=below, first_event_ok=first in ("glide_jump", "split"), no_preramp_move=pre,
                            stability=stab))


def test_acceptance_mapping():
    assert rpd.acceptance(_summ()) == (True, [])
    assert rpd.acceptance(_summ(below=False)) == (True, [])             # note c: recorded only
    assert rpd.acceptance(_summ(bracket=False, first=None, stab=False))[1] == ["no_bracket"]
    assert rpd.acceptance(_summ(bracket=False, pre=False, first="split", stab=False))[1] == ["no_bracket (moved during pre-ramp)"]
    assert rpd.acceptance(_summ(first="anomaly"))[1][0].startswith("anomaly")
    assert rpd.acceptance(_summ(stab=False))[1] == ["failed_seed"]


def _fake_env(tmp_path, monkeypatch, summaries, calib_ok=lambda chi, syz: True):
    monkeypatch.setattr(rpd, "CALIB_CSV", str(tmp_path / "calibrations.csv"))
    calls = dict(ramp=0, calib=[])

    def calibrate(chi, syz, tag):
        calls["calib"].append((chi, syz))
        return [{"gate_A3": "PASS"}], {f"q{p}": [0.0] * 5 for p in range(5)}, calib_ok(chi, syz)

    def ramp(chi, syz, runner, model, states, work, say, rn=None, levels=None, entry_fn=None):
        calls["ramp"] += 1
        return summaries[calls["ramp"] - 1]

    paths = dict(model_path=str(tmp_path / "model.json"), results=str(tmp_path / "res.csv"), states=str(tmp_path / "st.csv"),
                 work=str(tmp_path / "w"))
    return calls, calibrate, ramp, paths


def test_invalid_count_stop_T(tmp_path, monkeypatch):
    jobs = [j for j in TJ if abs(j["chi"]) == 10][:6]
    summ = [_summ(first="anomaly")] * 6
    calls, cal, ramp, paths = _fake_env(tmp_path, monkeypatch, summ)
    code = rpd.run_points("T", jobs, None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lambda m: None, **paths)
    assert code == 3 and calls["ramp"] == 3                            # stops at the 3rd invalid (> 2 of 49)
    assert rpd.run_points("T", jobs, None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lambda m: None, **paths) == 3
    assert calls["ramp"] == 3                                          # a restart does not run on
    assert not rpd.stop_reached("P", 18) and rpd.stop_reached("P", 19)
    assert not rpd.stop_reached("T", 2) and rpd.stop_reached("T", 3)


def test_gate_a_fail_is_invalid_without_ramp_and_calibrates_levels(tmp_path, monkeypatch):
    job = next(j for j in PJ if j["syz"] == 0.46875)
    calls, cal, ramp, paths = _fake_env(tmp_path, monkeypatch, [], calib_ok=lambda chi, syz: syz != 0.28125)
    code = rpd.run_points("P", [job], None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lambda m: None, **paths)
    assert code == 0 and calls["ramp"] == 0
    assert [s for _, s in calls["calib"]] == [0.09375, 0.28125, 0.46875]
    r = list(csv.DictReader(open(paths["results"])))[0]
    assert r["valid"] == "no" and r["reason"].startswith("gate_a_fail") and "+0.28" in r["reason"]
    m = json.load(open(paths["model_path"]))["correction"]
    assert r4.model_key(job["chi"], 0.28125) not in m and r4.model_key(job["chi"], 0.46875) in m
    assert "-30,+0.75" in m                                            # the step-3 entries are carried over unchanged


def test_valid_point_row_and_resume(tmp_path, monkeypatch):
    jobs = [j for j in TJ if j["chi"] == 20.0 and j["syz"] in (0.0, 0.25)]
    calls, cal, ramp, paths = _fake_env(tmp_path, monkeypatch, [_summ(), _summ(first="split")])
    assert rpd.run_points("T", jobs, None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lambda m: None, **paths) == 0
    rows = list(csv.DictReader(open(paths["results"])))
    assert [r["valid"] for r in rows] == ["yes", "yes"] and rows[1]["event"] == "split"
    assert rpd.run_points("T", jobs, None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lambda m: None, **paths) == 0
    assert calls["ramp"] == 2 and len(calls["calib"]) == 2             # nothing rerun


def test_pre_run_problem_stops_unscored(tmp_path, monkeypatch):
    job = next(j for j in TJ if j["chi"] == -10.0 and j["syz"] == 0.0)
    calls, cal, ramp, paths = _fake_env(tmp_path, monkeypatch, [_summ()])
    code = rpd.run_points("T", [job], None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, ["bad --"]), say=lambda m: None, **paths)
    assert code == 2 and calls["ramp"] == 0 and not os.path.exists(paths["results"])


def test_sealed_T_path_only_in_comparison_script():
    allowed = {os.path.join(ROOT, "scripts/v02_compare.py")}            # the comparison script (item 1), not written yet
    pat = re.compile(r"sealed|T_results")
    files = glob.glob(os.path.join(ROOT, "src/surrogate/**/*.py"), recursive=True) + glob.glob(os.path.join(ROOT, "scripts/**/*.py"), recursive=True)
    assert files
    offenders = [f for f in files if f not in allowed and pat.search(open(f).read())]
    assert not offenders, offenders


def test_stab_argv_default_model_unchanged():
    a = r4.stability_parser().parse_args(r4.stab_argv("x.data", 1.0, -30, "t", 1, 7, "-30,+0.75", 0.75))
    assert a.model == r4.MODEL
    b = r4.stability_parser().parse_args(r4.stab_argv("x.data", 1.0, -29.333333333333332, "t", 1, 7, "-29.3333,-0.66", -0.65625,
                                                      model_path=rpd.PROD_MODEL))
    assert b.model == rpd.PROD_MODEL and b.chi == -29.333333333333332 and b.syz == -0.65625 and b.model_key == "-29.3333,-0.66"


def test_T_output_is_blind(tmp_path, monkeypatch):
    """D-v02-9 item 3: for --set T no tau_P, bracket or event value is printed (progress lines, ramp state lines, summary)."""
    jobs = [j for j in TJ if j["chi"] == 10.0][:3]
    summ = [_summ(), _summ(first="split"), _summ(first="anomaly")]
    calls, cal, _, paths = _fake_env(tmp_path, monkeypatch, summ)

    def ramp(chi, syz, runner, model, states, work, say, rn=None, levels=None, entry_fn=None):
        calls["ramp"] += 1
        say(f"{rn} state 5 coarse syz {syz:+.2f} tau 1.23 | glide_jump | elapsed 1 s")    # what ramp_v02 prints per state
        return summ[calls["ramp"] - 1]

    out = []
    for set_, js in (("T", jobs), ("P", PJ[:3])):
        lines = []
        calls["ramp"] = 0
        p = {**paths, "results": str(tmp_path / f"res_{set_}.csv")}
        assert rpd.run_points(set_, js, None, cal, ramp=ramp, checks=lambda *a, **k: (1, 2, []), say=lines.append, **p) == 0
        lines.append(rpd.counts_summary(set_, p["results"]))
        out.append("\n".join(lines))
    t_out, p_out = out
    for bad in ("1.005", "1.01", "1.0 ", "1.23", "tau", "glide", "split", "first event", "bracket"):
        assert bad not in t_out, bad
    assert "1 invalid" in t_out and "anomaly 1" in t_out and "3 done, 2 valid" in t_out
    assert "tau 1.23" in p_out and "first event anomaly" in p_out      # P is not blind
    rows = list(csv.DictReader(open(tmp_path / "res_T.csv")))
    assert rows[0]["tau_P_mid"] == "1.005" and rows[2]["reason"].startswith("anomaly (first event")   # values still recorded


def test_production_calibration_writes_nothing_in_gate_dir(tmp_path, monkeypatch):
    """D-v02-10 log-file note: a production calibration (fake LAMMPS) leaves checks/10_gateA_yz untouched; its logs land in
    CALIB_TMP / CALIB_DIR. Step 3's default (out_dir '.') is unchanged."""
    import calibrate_yz as cy, inspect
    gate = os.path.join(ROOT, "checks/10_gateA_yz")
    snap = lambda: {f: os.stat(os.path.join(dp, f)).st_mtime_ns for dp, _, fs in os.walk(gate) for f in fs if "__pycache__" not in dp}
    before = snap()

    def fake_run(args, check=True, **k):
        log = args[args.index("-log") + 1]
        v = {args[i + 1]: args[i + 2] for i in range(len(args)) if args[i] == "-var"}
        assert os.path.exists(v["datain"]) and not os.path.abspath(v["datain"]).startswith(gate)
        open(log, "w").write("INTERIOR_GPa x " + " ".join(f"s{n}=0.0" for n in cy.NAMES) + "\n")

    def fake_calibrate(chi, syz, measure, K, log=print):
        measure(f"yz_chi{chi:+g}_syz{syz:+g}_tau0.5_it0", np.zeros(3), 0.1, syz)
        return [{"gate_A3": "PASS"}], {}, True

    monkeypatch.setattr(cy.subprocess, "run", fake_run)
    monkeypatch.setattr(cy, "calibrate", fake_calibrate)
    monkeypatch.setattr(rpd, "CALIB_TMP", str(tmp_path / "tmp"))
    monkeypatch.setattr(rpd, "CALIB_DIR", str(tmp_path / "calib"))
    cwd = os.getcwd()
    rpd.LammpsCalibrator("7")(-20.0, 0.25, "-20_+0.25")
    assert os.getcwd() == cwd and snap() == before
    assert os.path.exists(tmp_path / "tmp" / "log.yz_ref")
    assert \
        sorted(os.listdir(tmp_path / "calib" / "-20_+0.25")) == ["log.yz_chi-20_syz+0.25_tau0.5_it0", "runB.csv"]
    assert inspect.signature(cy.lammps_measure).parameters["out_dir"].default == "."


def test_P_night_kill_during_point2_and_resume(tmp_path, monkeypatch):
    """D-v02-12 acceptance: a P night batch killed partway through point 2 (after some of its P_states rows are written), then
    the same call again, against an uninterrupted run of the same fake (fake LAMMPS, real ramp_v02.run_ramp): (i) all result
    and state rows of points 1-3 identical, every field; only wall-clock fields (finished, seconds, ramp_seconds,
    calib_seconds) are excluded, and the folder prefix of the 'data' path is normalised; (ii) the rerun starts at the first
    state without a saved row; (iii) no duplicate or orphan rows; (iv) P_night1.json unchanged."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_v02 import _RampFake
    ids = rpd.night_batch(1, PJ, set(), prod=str(tmp_path))
    night_file = (tmp_path / "P_night1.json").read_bytes()
    by_id = {j["id"]: j for j in PJ}
    jobs = [by_id[i] for i in ids[:3]]
    thr = {(j["chi"], j["syz"]): 1.315 for j in jobs}
    rn = [rpd.run_name(j) for j in jobs]
    CLOCK = {"finished", "seconds", "ramp_seconds", "calib_seconds"}

    def env(sub):
        d = tmp_path / sub; d.mkdir()
        _, cal, _, paths = _fake_env(d, monkeypatch, [])
        return cal, paths, dict(checks=lambda *a, **k: (1, 2, []), say=lambda m: None, night=1, **paths)

    def rows(paths, which, root):
        return [{k: (v.replace(str(root), "<root>") if k == "data" else v) for k, v in r.items() if k not in CLOCK}
                for r in csv.DictReader(open(paths[which]))]

    cal_a, paths_a, kw_a = env("uninterrupted")                        # reference run
    assert rpd.run_points("P", jobs, _RampFake(thr), cal_a, **kw_a) == 0

    class Killed(Exception):
        pass

    class KillIn2(_RampFake):
        def relax(self, rdir, tag, *a):
            if os.path.basename(rdir) == rn[1] and "coarse" in tag and "tau0.20" in tag:
                raise Killed()
            return super().relax(rdir, tag, *a)

    cal_b, paths_b, kw_b = env("killed")
    with pytest.raises(Killed):
        rpd.run_points("P", jobs, KillIn2(thr), cal_b, **kw_b)
    st1 = list(csv.DictReader(open(paths_b["states"])))
    n2_saved = sum(r["run"] == rn[1] for r in st1)
    assert [r["id"] for r in csv.DictReader(open(paths_b["results"]))] == [jobs[0]["id"]] and n2_saved > 0
    fake = _RampFake(thr)
    assert rpd.run_points("P", jobs, fake, cal_b, **kw_b) == 0
    # (i) every row identical to the uninterrupted run
    for which in ("results", "states"):
        a, b = rows(paths_a, which, tmp_path / "uninterrupted"), rows(paths_b, which, tmp_path / "killed")
        assert len(a) == len(b), which
        for ra, rb in zip(a, b):
            assert ra == rb, (which, {k: (ra[k], rb.get(k)) for k in ra if ra[k] != rb.get(k)})
    # (ii) the rerun's first LAMMPS call is point 2's first state without a saved row (point 1 not touched)
    assert fake.calls[0].startswith(f"{n2_saved:02d}_")
    assert sum(r["run"] == rn[0] for r in csv.DictReader(open(paths_b["states"]))) == sum(r["run"] == rn[0] for r in st1)
    # (iii) no duplicates, no orphans
    st = list(csv.DictReader(open(paths_b["states"]))); res = list(csv.DictReader(open(paths_b["results"])))
    keys = [(r["run"], r["idx"], r["phase"]) for r in st]
    assert len(keys) == len(set(keys)) and [r["id"] for r in res] == [j["id"] for j in jobs]
    assert {r["run"] for r in st} == set(rn)
    # (iv) the night batch file is unchanged
    assert (tmp_path / "P_night1.json").read_bytes() == night_file
