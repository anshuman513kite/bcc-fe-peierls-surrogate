"""v0.2 step-2 tests (2 Nov 2026). No LAMMPS process is started by any test (the memory-sampler test watches a dummy process).
sigma_yz loading (input regression, command, pre-ramp ladder, strain-model correction, surface force), three-component Gate A,
the Gate A-yz calibration logic on a fake slab, grid queue kill/resume, memory sampler, two-GP surrogate, designs, synthetic
surfaces and the pool-size choice rule."""
import json, os, re, signal, subprocess, sys, time
import numpy as np
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in ("tools", "src", "scripts", "checks/10_gateA_yz", "checks/11_core_gate", "checks/12_corner_gate", "checks/13_size_gate"):
    sys.path.insert(0, os.path.join(ROOT, p))
from loading_v02 import step_cmd, syz_ladder, model_correction, syz_correction, surface_force
from gate_a import gate_a3_check
from surrogate.event_gp import EventGP, scale_x
from surrogate.designs import grid_T, make_pool, random_order, spacefill_order, grid_designs, nested_1d, T_SIG
from surrogate.synthetic2d import truth


# ---------------- sigma_yz loading ----------------
from render_lmp import render as _render, cmd_vars, bad_numbers      # 3 Nov: shared renderer (also substitutes inside index defaults)


V01_VARS = dict(pot="P", datain="in.data", dataout="out.data", peout="out.pe", fx="0.0179910000", y0="74.0301", Lx="14.8366",
                Lz="146.8746", xz="1.2364", ys="74.03", zs="72.27", maxit="10000", groups="groups.lmp", halt_dist="6.9939",
                etol="0", ftol="1e-6", dmax="0.01", maxeval="2000000")


def test_step_input_identical_to_v01_at_syz_zero():
    old = _render(os.path.join(ROOT, "tests/data/in.step_chi_v01"), V01_VARS)
    new = _render(os.path.join(ROOT, "tools/in.step_chi"), V01_VARS)
    extra = [l for l in new if re.match(r"variable\s+(fzb?\s+index\s+0\.0|fxb\s+index\s+-0\.0179910000)$", l)]
    assert len(extra) == 3                                   # the only additions: default declarations of fz, fzb, fxb
    assert [l for l in new if l not in extra] == old         # every other evaluated command identical, fix lines included
    fix = [re.sub(r"\s+", " ", l) for l in new if "addforce" in l]
    assert fix == ["fix ftop top addforce 0.0179910000 0.0 0.0", "fix fbot bot addforce -0.0179910000 0.0 0.0"]


def test_step_input_negative_syz_has_no_double_minus():
    new = _render(os.path.join(ROOT, "tools/in.step_chi"), dict(V01_VARS, fz="-0.0050000000", fzb="0.0050000000"))
    fix = [re.sub(r"\s+", " ", l) for l in new if "addforce" in l]
    assert fix == ["fix ftop top addforce 0.0179910000 0.0 -0.0050000000", "fix fbot bot addforce -0.0179910000 0.0 0.0050000000"]


def _v01_cmd(np_, root, tag, datain, fx, y0, Lx, Lz, xz, ys, zs, maxit, halt, etol, ftol, dmax, maxeval):
    """The v0.1 command, copied verbatim from tools/peierls_chi.py run_step() at commit a539f18."""
    POT = f"{root}/potentials/Fe_mm.eam.fs"
    return ["mpirun", "-np", str(np_), "lmp", "-in", f"{root}/tools/in.step_chi", "-screen", "none", "-log", f"steps/log.{tag}",
            "-var", "pot", POT, "-var", "datain", datain, "-var", "dataout", f"steps/{tag}.data", "-var", "peout", f"steps/{tag}.pe",
            "-var", "fx", f"{fx:.10f}", "-var", "y0", f"{y0}", "-var", "Lx", f"{Lx:.10f}", "-var", "Lz", f"{Lz:.10f}",
            "-var", "xz", f"{xz:.10f}", "-var", "ys", f"{ys:.6f}", "-var", "zs", f"{zs:.6f}", "-var", "maxit", str(maxit),
            "-var", "groups", "groups.lmp", "-var", "halt_dist", f"{halt:.4f}",
            "-var", "etol", etol, "-var", "ftol", ftol, "-var", "dmax", dmax, "-var", "maxeval", str(maxeval)]


def test_step_cmd_identical_to_v01_at_syz_zero():
    r, tag = "/R", "coarse_1.250"
    a = (7, r, tag, "prev.data", 0.0224887, 74.0301, 14.83660001, 146.87, 1.23, 74.1, 72.3, 200000, 6.99393, "0", "1e-6", "0.01", 2000000)
    old = _v01_cmd(*a)
    new = step_cmd(7, f"{r}/tools/in.step_chi", f"steps/log.{tag}", f"{r}/potentials/Fe_mm.eam.fs", "prev.data", f"steps/{tag}.data",
                   f"steps/{tag}.pe", 0.0224887, 74.0301, 14.83660001, 146.87, 1.23, 74.1, 72.3, 200000, "groups.lmp", 6.99393,
                   "0", "1e-6", "0.01", 2000000)
    assert new == old
    with_fz = step_cmd(7, f"{r}/tools/in.step_chi", f"steps/log.{tag}", f"{r}/potentials/Fe_mm.eam.fs", "prev.data", f"steps/{tag}.data",
                       f"steps/{tag}.pe", 0.0224887, 74.0301, 14.83660001, 146.87, 1.23, 74.1, 72.3, 200000, "groups.lmp", 6.99393,
                       "0", "1e-6", "0.01", 2000000, fz=-0.009)
    assert with_fz == old + ["-var", "fz", "-0.0090000000", "-var", "fzb", "0.0090000000"]


def test_syz_ladder():
    assert syz_ladder(0.0) == []
    assert syz_ladder(1.0) == [0.25, 0.5, 0.75, 1.0]
    assert syz_ladder(0.25) == [0.25]
    lad = syz_ladder(-0.6)
    assert lad == [-0.2, -0.4, -0.6]
    for t in (0.1, 0.33, 0.5, 0.74, -1.0, 0.999):
        lad = syz_ladder(t)
        steps = np.diff([0.0] + lad)
        assert abs(lad[-1] - t) < 1e-9 and np.all(np.abs(steps) <= 0.25 + 1e-9)


def test_model_correction_v01_entry_unchanged():
    q2, q3, q4 = [1e-4, -2e-4, 3e-4, 1e-3], [-1e-5, 2e-5, 0.0, -2e-4], [3e-6, 0.0, -1e-6, 5e-5]
    entry = dict(q2=q2, q3=q3, q4=q4)
    for tau in (0.0, 0.5, 1.315, 2.4):
        Q = np.array([q2, q3, q4])
        v01 = Q[0] * tau ** 2 + Q[1] * tau ** 3 + Q[2] * tau ** 4        # the v0.1 expression
        c = model_correction(entry, tau)
        assert np.array_equal(c, v01) and syz_correction(c) == 0.0
    e5 = dict(q0=[0, 0, 0, 0, 0.001], q1=[0, 0, 0, 0, 0.002], q2=[0, 0, 0, 0, 0.0])
    assert abs(syz_correction(model_correction(e5, 2.0)) - 0.005) < 1e-12


def test_surface_force_session5_hand_check():
    assert abs(surface_force(1.0, 14.8366, 146.8746, 756) - 0.017991) < 2e-6     # Session 5: 0.017991 eV/A per atom per GPa


# ---------------- three-component Gate A ----------------
def _s(chi, tau, syz, d=None):
    s = np.zeros(6); s[5] = tau * np.cos(np.radians(chi)); s[4] = tau * np.sin(np.radians(chi)); s[3] = syz
    if d:
        for k, v in d.items():
            s[k] += v
    return s


def test_gate_a3_exact_loading_passes():
    for chi, tau, syz in ((-30, 2.4, 1.0), (0, 1.0, -0.5), (30, 3.0, 0.25), (12, 0.5, 0.0)):
        assert gate_a3_check(_s(chi, tau, syz), chi, tau, syz)["ok"]


def test_gate_a3_syz_target_zero_uses_floor():
    # tau = 1, syz = 0: floor 0.001745 * 1 GPa
    assert gate_a3_check(_s(0, 1.0, 0.0, {3: 0.0015}), 0, 1.0, 0.0)["ok"]
    assert not gate_a3_check(_s(0, 1.0, 0.0, {3: 0.0020}), 0, 1.0, 0.0)["ok"]


def test_gate_a3_syz_relative_and_floor():
    assert gate_a3_check(_s(0, 1.0, 1.0, {3: 0.009}), 0, 1.0, 1.0)["ok"]           # 0.9 % of target
    assert not gate_a3_check(_s(0, 1.0, 1.0, {3: 0.012}), 0, 1.0, 1.0)["ok"]       # 1.2 % and above the floor 0.001745
    g = gate_a3_check(_s(0, 2.0, 0.1, {3: 0.0030}), 0, 2.0, 0.1)                   # 3 % of a small target, but under 0.001745 * 2
    assert g["ryz"] > 0.01 and g["ok"]


def test_gate_a3_d2_branch_for_sigma_xz():
    g = gate_a3_check(_s(3, 0.5, 0.25, {4: 0.0003}), 3, 0.5, 0.25)
    assert g["rxz"] > 0.01 and g["ok_xz"] and g["ok"]                               # v0.1 chi = +3 case: passes by D2
    assert not gate_a3_check(_s(3, 0.5, 0.25, {4: 0.01}), 3, 0.5, 0.25)["ok"]


def test_gate_a3_tau_zero_points():
    syz = 0.5                                       # floor 0.001745 * max(0, 0.5, 1.0) = 0.001745 GPa (V02_PLAN 8.8; was 0.00087)
    assert gate_a3_check(_s(30, 0.0, syz, {5: 0.0008, 4: -0.0008}), 30, 0.0, syz)["ok"]
    assert gate_a3_check(_s(30, 0.0, syz, {5: 0.0010}), 30, 0.0, syz)["ok"]        # 8.8: below the 0.001745 floor now (failed before 8.8)
    assert not gate_a3_check(_s(30, 0.0, syz, {5: 0.0020}), 30, 0.0, syz)["ok"]    # sigma_xy leftover above the floor
    assert gate_a3_check(_s(30, 0.0, syz, {0: 0.009}), 30, 0.0, syz)["ok"]          # sigma_xx 1.8 % of |syz|
    assert not gate_a3_check(_s(30, 0.0, syz, {2: 0.011}), 30, 0.0, syz)["ok"]      # sigma_zz 2.2 % of |syz|
    with pytest.raises(ValueError):
        gate_a3_check(np.zeros(6), 0, 0.0, 0.0)


def test_gate_a3_others_relative_to_tau_when_loaded():
    assert not gate_a3_check(_s(0, 1.0, 1.0, {1: 0.025}), 0, 1.0, 1.0)["ok"]      # 2.5 % of tau (not of |syz|)


# ---------------- Gate A-yz calibration logic on a fake slab ----------------
def test_calibrate_yz_on_fake_slab():
    from calibrate_yz import calibrate, slab_stiffness
    K = slab_stiffness()

    def fake(tag, e, sxy, syz):                       # linear slab + second-order cross terms the calibration must remove
        tau = float(re.search(r"tau([0-9.]+)", tag).group(1))
        box = K @ e + np.array([0.004 * sxy ** 2, -0.003 * syz ** 2, 0.002 * sxy * syz])
        return np.array([box[0], 0.0, box[1], syz + 0.02 * sxy * syz, box[2], sxy + 0.01 * sxy ** 2])

    for chi, syz in ((-30, 1.0), (0, -0.5), (30, 0.0)):
        rows, entry, ok = calibrate(chi, syz, fake, K, log=lambda m: None)
        assert ok, [r for r in rows if r["gate_A3"] != "PASS"]
        assert ("q0" in entry) == (syz != 0.0)
        assert any(r["tau"] == 0.0 for r in rows) == (syz != 0.0)
        assert len(entry["q2"]) == 5


# ---------------- grid queue: kill and resume ----------------
def test_grid_queue_kill_and_resume(tmp_path):
    runs = tmp_path / "runs.txt"
    jobs = [dict(id=f"p{i}", cwd=str(tmp_path),
                 cmd=[sys.executable, "-c", f"import time; open('runs.txt','a').write('p{i}\\n'); time.sleep(0.8); open('p{i}.ok','w').write('1')"])
            for i in range(5)]
    jf, sf = tmp_path / "jobs.json", tmp_path / "status.json"
    jf.write_text(json.dumps(jobs))
    q = [sys.executable, os.path.join(ROOT, "tools/grid_queue.py"), str(jf), str(sf), "pytest_queue", "--logdir", str(tmp_path)]
    p = subprocess.Popen(q, start_new_session=True, stdout=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 20:
        st = json.loads(sf.read_text()) if sf.exists() else {}
        if (sum(v["state"] == "done" for v in st.values()) == 2 and any(v["state"] == "running" for v in st.values())
                and runs.exists() and "p2" in runs.read_text().split()):
            break
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGKILL); p.wait()
    st = json.loads(sf.read_text())                                   # status file is never half-written
    assert [st[f"p{i}"]["state"] for i in range(5)] == ["done", "done", "running", "pending", "pending"]
    assert not (tmp_path / "p2.ok").exists()
    rc = subprocess.run(q, stdout=subprocess.DEVNULL).returncode
    st = json.loads(sf.read_text())
    assert rc == 0 and all(v["state"] == "done" for v in st.values())
    assert runs.read_text().split() == ["p0", "p1", "p2", "p2", "p3", "p4"]   # done jobs not re-run; the killed one re-run
    assert "resume: 1 job(s)" in (tmp_path / "pytest_queue.log").read_text()


# ---------------- memory sampler (dummy process, never lmp) ----------------
def test_memory_sampler_on_dummy_process(tmp_path):
    dummy = tmp_path / "lmpdummy"
    os.symlink(sys.executable, dummy)                                  # process name 'lmpdummy', not lmp
    name = f"pytest_memsampler_{os.getpid()}"
    env = dict(os.environ, MEM_PROC="lmpdummy", MEM_INTERVAL="1")
    code = "x = bytearray(150 * 2**20); import time; time.sleep(4)"
    subprocess.run([os.path.join(ROOT, "tools/launch_safe.sh"), name, str(dummy), "-c", code], env=env, check=True,
                   stdout=subprocess.DEVNULL, cwd=tmp_path)
    log, pidf = os.path.join(ROOT, "logs", f"{name}.log"), os.path.join(ROOT, "logs", f"{name}.pid")
    try:
        pid = int(open(pidf).read())
        t0 = time.time()
        while time.time() - t0 < 20 and not (os.path.exists(log) and "MEM peak" in open(log).read()):
            time.sleep(0.2)
        text = open(log).read()
        lines = [l for l in text.splitlines() if " MEM lmpdummy:" in l]
        assert len(lines) >= 2 and all("processes=1 " in l for l in lines)
        peak = int(re.search(r"MEM peak .*: (\d+) MB", text).group(1))
        assert 150 <= peak < 400, peak                                  # 150 MB array + interpreter
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        for ext in ("log", "pid", "out"):
            f = os.path.join(ROOT, "logs", f"{name}.{ext}")
            if os.path.exists(f):
                os.remove(f)


# ---------------- two-GP surrogate ----------------
def test_event_gp_split_prior_cannot_undercut_far_from_split_data():
    rng = np.random.default_rng(0)
    chi = rng.uniform(-30, 30, 25); sig = rng.uniform(-1, 1, 25)
    y, ev = truth(chi, sig, "medium")
    m = EventGP(0).fit(scale_x(chi, sig), y, ev)
    assert m.n_glide == (ev == "glide").sum() and m.n_split == (ev == "split").sum()
    # glide-only training data: the split surface is its prior (highest observed tau_P) everywhere -> never the lower one
    g = ev == "glide"
    m2 = EventGP(0).fit(scale_x(chi[g], sig[g]), y[g], ev[g])
    Xq = scale_x(np.linspace(-30, 30, 13).repeat(5), np.tile(np.linspace(-1, 1, 5), 13))
    (mg, _), (ms, _) = m2.predict_both(Xq)
    assert np.allclose(ms, y[g].max()) and np.all(m2.predict(Xq)[2] == np.where(mg <= ms, "glide", "split"))
    # two split points at chi = -30: far away (chi >= 0) the split mean stays near the highest observed value
    chi3 = np.r_[chi[g], -30, -30]; sig3 = np.r_[sig[g], -0.5, 0.5]
    y3, ev3 = truth(chi3, sig3, "medium")
    m3 = EventGP(0).fit(scale_x(chi3, sig3), y3, ev3)
    far = scale_x(np.linspace(0, 20, 5), np.zeros(5))
    (mg3, _), (ms3, _) = m3.predict_both(far)
    assert np.all(ms3 > mg3 + 0.1) and np.all(m3.predict(far)[2] == "glide")   # no undercut where the split GP has no data


def test_event_gp_spread_and_mean():
    chi = np.array([-30, -30, 0, 0, 30, 30, -10, 10.0]); sig = np.array([-1, 1, -1, 1, -1, 1, 0, 0.0])
    y, ev = truth(chi, sig, "weak")
    m = EventGP(1).fit(scale_x(chi, sig), y, ev)
    mu, sp, pev = m.predict(scale_x(chi, sig))
    assert np.sqrt(np.mean((mu - y) ** 2)) < 0.02 and np.all(sp > 0)


# ---------------- designs ----------------
def test_designs():
    T = grid_T()
    assert T.shape == (49, 2)
    for M in (60, 100, 150, 240, 2000):
        P, nc, ns = make_pool(M)
        assert ns % 2 == 0
        assert not any(np.any(np.all(np.isclose(T, p), axis=1)) for p in P)          # disjoint from T
        edge = P[nc * ns:]
        assert len(edge) == 2 * ns and set(edge[:, 0]) == {-30.0, 30.0}
        assert not any(np.any(np.isclose(T_SIG, v)) for v in edge[:, 1])           # edge levels not in T
        assert abs(len(P) - M) <= max(4, 0.05 * M)
        G = grid_designs(nc, ns)
        assert len(G[0]) == 4 and sorted(G[-1]) == list(range(nc * ns))
        assert all(set(a) < set(b) for a, b in zip(G[:-1], G[1:]))                  # nested
    assert sorted(nested_1d(9)) == list(range(9)) and nested_1d(9)[:3] == [0, 8, 4]
    P, _, _ = make_pool(100)
    assert spacefill_order(P, 3) == spacefill_order(P, 3) and sorted(spacefill_order(P, 3)) == list(range(len(P)))
    assert sorted(random_order(len(P), 3)) == list(range(len(P))) and random_order(len(P), 3) != random_order(len(P), 4)


def test_synthetic_surfaces_anchor_v01():
    y, ev = truth([-30, 0, 30, 27], [0, 0, 0, 0], "medium")
    assert ev.tolist() == ["split", "glide", "glide", "split"]
    assert np.allclose(y[:3], [1.155, 1.315, 2.055])     # bracket midpoints: 1.1599 -> 1.150-1.160, 1.3151, 2.0501
    assert np.allclose(np.round((y - 0.005) / 0.01), (y - 0.005) / 0.01)


def test_pool_size_choice_rule():
    from v02_pool_study import choose
    assert choose({3: (2.0, 0.0, 4.0), 5: (3.0, 1.0, 6.0), 8: (3.5, 1.0, 7.0)})[0] == 3   # inside the 3N interval: stops at 3N
    assert choose({3: (2.0, 0.0, 4.0), 5: (6.0, 4.5, 8.0), 8: (7.0, 5.0, 9.0)})[0] == 5   # grows to 5N, stops at 8N
    assert choose({3: (2.0, 0.0, 4.0), 5: (6.0, 5.0, 7.0), 8: (9.0, 8.0, 10.0)})[0] is None  # still growing at 8N
    assert choose({3: (2.0, 0.0, 4.0), 5: (4.0, 3.0, 5.0), 8: (4.5, 3.0, 6.0)})[0] == 5   # exactly on the upper end: borderline -> larger
    assert choose({3: (5.0, 3.0, 7.0), 5: (1.0, 0.0, 2.0), 8: (1.0, 0.0, 2.0)})[0] == 3   # fell below the interval: not growing


# ---------------- Q10 revised: event classifier (V02_PLAN 8.2) ----------------
def test_event_gpc_zero_split_points_predicts_glide():
    from surrogate.event_gp import EventGPC
    chi = np.linspace(-20, 20, 9).repeat(3); sig = np.tile([-1.0, 0.0, 1.0], 9)
    y, ev = truth(chi, sig, "weak")
    assert (ev == "glide").all()
    m = EventGPC(0).fit(scale_x(chi, sig), y, ev)
    Xq = scale_x(np.linspace(-30, 30, 13), np.zeros(13))
    assert m.clf is None and np.all(m.p_split(Xq) == 0.0) and np.all(m.predict(Xq)[2] == "glide")


def test_event_gpc_no_split_between_two_patches():
    from surrogate.event_gp import EventGPC
    chi = np.linspace(-30, 30, 31).repeat(3); sig = np.tile([-1.0, 0.0, 1.0], 31)
    y, ev = truth(chi, sig, "weak")                      # split only near chi = -30 and near +27: two separate patches
    sp = chi[ev == "split"]
    assert sp.min() == -30 and sp.max() > 24 and not np.any((sp > -20) & (sp < 20))
    m = EventGPC(0).fit(scale_x(chi, sig), y, ev)
    cq, sq = np.meshgrid(np.linspace(-15, 15, 13), np.linspace(-1, 1, 5))
    Xq = scale_x(cq.ravel(), sq.ravel())
    assert np.all(m.predict(Xq)[2] == "glide") and np.all(m.p_split(Xq) < 0.5)
    assert np.all(m.predict(scale_x([-30.0, 27.0], [0.0, 0.0]))[2] == "split")   # and split where the patches are


# ---------------- fix (d): min rule inside the classifier's split region (V02_PLAN 8.3) ----------------
def test_event_gpd_no_split_between_two_patches():
    from surrogate.event_gp import EventGPD
    chi = np.linspace(-30, 30, 31).repeat(3); sig = np.tile([-1.0, 0.0, 1.0], 31)
    y, ev = truth(chi, sig, "weak")
    m = EventGPD(0).fit(scale_x(chi, sig), y, ev)
    cq, sq = np.meshgrid(np.linspace(-15, 15, 13), np.linspace(-1, 1, 5))
    assert np.all(m.predict(scale_x(cq.ravel(), sq.ravel()))[2] == "glide")
    assert np.all(m.predict(scale_x([-30.0, 27.0], [0.0, 0.0]))[2] == "split")


def test_event_gpd_glide_wins_inside_split_region_when_lower():
    from surrogate.event_gp import EventGPD
    chi = np.linspace(-30, 30, 31).repeat(3); sig = np.tile([-1.0, 0.0, 1.0], 31)
    y, ev = truth(chi, sig, "weak")
    m = EventGPD(0).fit(scale_x(chi, sig), y, ev)
    X = scale_x([-30.0], [0.0])
    assert m.p_split(X)[0] > 0.5
    (mg, _), (ms, _) = m.predict_both(X)
    m.split.m0 += 1.0                                   # raise the split surface: classifier still says split, glide is lower
    (mg2, _), (ms2, _) = m.predict_both(X)
    assert m.p_split(X)[0] > 0.5 and mg2[0] < ms2[0]
    mean, spread, event = m.predict(X)
    assert event[0] == "glide" and mean[0] == mg2[0]


# ---------------- V02_PLAN 8.7: Gate A-yz ladder, required ramp ----------------
def test_gate_a_yz_ladders_8p7():
    from calibrate_yz import ladders, calibrate, slab_stiffness
    assert ladders(0.25) == ((0.0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0), (0.25, 0.75, 1.25, 1.75, 2.2, 2.55, 2.85))
    assert ladders(-1.0) == ladders(0.25)
    assert ladders(0.0) == ((1.0, 1.5, 2.0, 2.4, 2.7, 3.0), (0.5, 1.25, 1.75, 2.2, 2.55, 2.85))     # sigma_yz = 0 unchanged
    seen = []
    fake = lambda tag, e, sxy, syz: (seen.append(float(re.search(r"tau([0-9.]+)", tag).group(1))), np.zeros(6))[1]
    calibrate(0, 0.25, fake, slab_stiffness(), max_it=0, log=lambda m: None)
    assert sorted(set(seen)) == sorted(set(ladders(0.25)[0] + ladders(0.25)[1]))


def test_peierls_chi_requires_coarse(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/peierls_chi.py"), "x.data", "74.0", "72.0", "t", "--chi", "0"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and "--coarse" in r.stderr                    # argparse error before any file or LAMMPS work
    assert not (tmp_path / "steps").exists()


# ---------------- V02_PLAN 8.8: Gate A tolerance floor ----------------
def test_gate_a3_floor_8p8():
    g = gate_a3_check(_s(0, 0.25, 0.25, {4: 0.00104}), 0, 0.25, 0.25)              # (a) S1's tau = 0.25 check: passes now
    assert g["ok"] and abs(g["floor"] - 0.001745) < 1e-6
    ok = lambda d: gate_a3_check(_s(0, 2.0, 0.25, {4: d}), 0, 2.0, 0.25)
    assert ok(0.0034)["ok"] and not ok(0.0036)["ok"]                              # (b) tau = 2.0: tolerance 0.00349, as before
    assert abs(ok(0.0)["floor"] - 0.001745 * 2.0) < 1e-6                          # radians(0.1) = 0.00174533
    assert not gate_a3_check(_s(0, 0.25, 0.25, {4: 0.0020}), 0, 0.25, 0.25)["ok"]  # (c) 0.0020 > 0.001745: fails
    assert not gate_a3_check(_s(0, 0.25, 0.25, {3: 0.0030}), 0, 0.25, 0.25)["ok"]  # sigma_yz: 1.2 % and above the floor: fails
    assert gate_a3_check(_s(0, 0.0, 0.25, {5: 0.0015}), 0, 0.0, 0.25)["ok"]        # tau = 0: sigma_xy held to 0.001745 too


# ---------------- step-3 driver (V02_PLAN 8.9), fake measurement only ----------------
def _fake_slab():
    from calibrate_yz import slab_stiffness
    K = slab_stiffness()

    def fake(tag, e, sxy, syz):
        box = K @ e + np.array([0.004 * sxy ** 2, -0.003 * syz ** 2, 0.002 * sxy * syz])
        return np.array([box[0], 0.0, box[1], syz + 0.02 * sxy * syz, box[2], sxy + 0.01 * sxy ** 2])
    return fake, K


def test_step3_job_list():
    import run_step3 as r3
    jobs = r3.read_jobs(os.path.join(ROOT, "checks/10_gateA_yz/jobs_step3.txt"))
    assert len(jobs) == 27 and len(set(jobs)) == 27
    assert set(jobs) == {(c, s) for c in (-30.0, 0.0, 30.0) for s in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)}


def test_step3_dry_run_touches_no_lammps(monkeypatch, capsys):
    import run_step3 as r3, calibrate_yz
    def boom(*a, **k):
        raise AssertionError("LAMMPS or a subprocess was started in the dry run")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(calibrate_yz, "lammps_measure", boom)
    cwd = os.getcwd()
    try:
        assert r3.main(["--dry-run"]) == 0
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert out.count("Run B: calibrate tau") == 27 and "identical: {-30: True, 0: True, 30: True}" in out


def test_step3_run_b_every_pair_and_resume(tmp_path):
    import run_step3 as r3
    from calibrate_yz import calibrate
    fake, K = _fake_slab()
    calls = []
    def calib(chi, syz, measure, K, log=print):
        calls.append((chi, syz)); return calibrate(chi, syz, measure, K, log=log)
    jobs = r3.read_jobs(os.path.join(ROOT, "checks/10_gateA_yz/jobs_step3.txt"))
    kw = dict(results=str(tmp_path / "res.csv"), outdir=str(tmp_path / "step3"), strain_model=str(tmp_path / "sm.json"),
              say=lambda m: None, calib=calib, move_logs=False)
    r3.run_pairs(jobs[:4], fake, K, **kw)
    assert calls == jobs[:4]                                            # Run B for every pair (Run A passing or not)
    res = r3.run_pairs(jobs[:6], fake, K, **kw)                         # resume: the first 4 are skipped
    assert calls == jobs[:4] + jobs[4:6] and len(res) == 6
    assert len(open(tmp_path / "res.csv").read().splitlines()) == 1 + 6
    sm = json.load(open(tmp_path / "sm.json"))["correction"]
    assert set(sm) == {r3.key(c, s) for c, s in jobs[:6]}               # entries accumulate (all pass on this fake)
    assert (tmp_path / "step3" / "chi-30_syz-1" / "runA.csv").exists() and (tmp_path / "step3" / "chi-30_syz-1" / "runB.csv").exists()


def test_step3_verdict_logic():
    import run_step3 as r3
    pairs = [(c, s) for c in (-30.0, 0.0, 30.0) for s in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)]
    ok = {p: True for p in pairs}
    assert r3.step3_verdict(ok) == (1.0, True, "PASS")
    assert r3.step3_verdict({**ok, (30.0, -0.75): False}) == (0.5, True, "PASS")      # failing |syz| = 0.75 -> s_A = 0.5
    assert r3.step3_verdict({**ok, (0.0, 0.25): False})[2] == "FAIL"                # failing 0.25 pair -> no s_A
    assert r3.step3_verdict({**ok, (-30.0, 0.0): False}) == (1.0, False, "FAIL")    # failing syz = 0 pair -> FAIL
    assert r3.step3_verdict({p: True for p in pairs[:-1]})[2] == "INCOMPLETE"


def test_step3_al_identity_check(tmp_path):
    import run_step3 as r3
    assert r3.al_identity() == {-30: True, 0: True, 30: True}
    m = json.load(open(r3.MODEL08)); m["correction"]["0"]["q2"][0] += 1e-9
    p = tmp_path / "al.json"; p.write_text(json.dumps(m))
    assert r3.al_identity(path08=str(p)) == {-30: True, 0: False, 30: True}


def test_step3_run_a_loading_uses_v01_entry_and_plain_syz():
    import run_step3 as r3
    entry = json.load(open(r3.MODEL07))["correction"]["30"]
    e, sxy, sy = r3.run_a_loading(entry, 30.0, 0.0, 0.75)
    assert np.allclose(e, 0) and sxy == 0 and sy == 0.75                   # v0.1 form: zero correction at tau = 0
    e, sxy, sy = r3.run_a_loading(entry, 30.0, 2.0, -0.5)
    Q = np.array([entry["q2"], entry["q3"], entry["q4"]]); c = Q[0] * 4 + Q[1] * 8 + Q[2] * 16
    assert np.allclose(e, r3.LIN * 2.0 * np.sin(np.radians(30)) + c[:3]) and np.isclose(sxy, 2.0 * np.cos(np.radians(30)) + c[3])
    assert sy == -0.5                                                     # plain sigma_yz, no correction


# ---------------- step-4 driver (V02_PLAN 8.10), fakes only ----------------
class _FakeRunner:
    def __init__(self, fail=None):
        self.fail, self.calls = fail or set(), []

    def setup(self, rdir):
        os.makedirs(rdir, exist_ok=True)

    def relax(self, rdir, tag, datain, entry, chi, tau, syz):
        self.calls.append((rdir, tag, round(syz, 2), tau))
        bad = (os.path.basename(rdir), int(tag[:2])) in self.fail
        return dict(data=os.path.join(rdir, tag), pe=-100.0, fmax=1e-8, stop="force tolerance",
                    dd=dict(dy=2.019 if bad else 0.0, dz=1.166 if bad else 0.0, orient="down", topcol=1.2, col=(0.67, 1.17)))

    def stability(self, rdir, tag, data, key, chi, tau, syz, seed):
        return True, "returned=yes"


def _fake_cell(path):
    r = np.zeros((4, 3)); r[:, 0] = [0, 1, 2, 3]
    return dict(Lx=10.0, Lz=10.0, xz=0.0, ids=np.arange(4), r=r)


def test_step4_state_list_and_order():
    import run_step4 as r4
    st = r4.states(30, -1)
    assert len(st) == 33 and [s["idx"] for s in st] == list(range(33))
    assert [s["phase"] for s in st[:4]] == ["start", "ladder", "ladder", "ladder"] and [s["syz"] for s in st[:4]] == [0.0, -0.25, -0.5, -0.75]
    assert [s["tau"] for s in st[4:16]] == [round(0.05 * k, 2) for k in range(1, 13)] and all(s["syz"] == -0.75 for s in st[4:16])
    assert [s["phase"] for s in st[16:18]] == ["stab1", "stab2"]
    assert [s["tau"] for s in st[18:30]] == [round(0.05 * k, 2) for k in range(11, 0, -1)] + [0.0]
    assert [s["syz"] for s in st[30:]] == [-0.5, -0.25, 0.0] and str(st[-1]["syz"]) == "0.0"       # not -0.0
    assert [s["level"] for s in st] == [0.0, 0.25, 0.5] + [0.75] * 30
    assert sum(len(r4.states(c, g)) for c in r4.CHIS for g in r4.SIGNS) == 6 * 33


def test_step4_loading_uses_step3_entry():
    import run_step4 as r4
    from loading_v02 import model_correction
    model = json.load(open(r4.MODEL))
    for chi, syz in ((-30, 0.5), (0, -0.75), (30, 0.25), (0, 0.0), (30, -0.0)):
        e = r4.entry_for(model, chi, syz or 0.0)
        assert e is model["correction"][f"{chi:g},{(syz or 0.0):+.2f}"]
    e = r4.entry_for(model, -30, 0.5)
    ee, sxy, dsyz = r4.loading(e, -30, 0.4)
    c = model_correction(e, 0.4)
    assert np.allclose(ee, r4.LIN * 0.4 * np.sin(np.radians(-30)) + c[:3]) and np.isclose(dsyz, c[4])
    with pytest.raises(KeyError):
        r4.entry_for(model, 30, -1.0)                      # failed in step 3: no entry, never used (s_A = 0.75)


def test_step4_verdict_logic(tmp_path):
    import run_step4 as r4
    model = json.load(open(r4.MODEL))
    def go(fail):
        res = tmp_path / f"r{len(list(tmp_path.iterdir()))}.csv"
        rows = r4.run_all(_FakeRunner(fail), model, results=str(res), work=str(tmp_path / "w"), say=lambda m: None, read=_fake_cell)
        return r4.verdict(rows)
    assert go(set()) == (0.75, 0.75, "PASS", "")
    # D-v02-5 (b): a fallback level is not granted without the full sequence -> INCOMPLETE, follow-up named (was PASS at 0.5)
    v = go({("chi+30_m", 3)})                                 # failure at the 0.75 ladder state -> s_4 = 0.5
    assert v[:3] == (0.5, None, "INCOMPLETE") and "follow-up at 0.5" in v[3]
    assert go({("chi+0_p", 1)})[2] == "FAIL"                 # failure at 0.25 -> FAIL
    v = go({("chi-30_p", 25)})                                # unload failure fails that run's top level
    assert v[:3] == (0.5, None, "INCOMPLETE") and "follow-up at 0.5" in v[3]
    assert go({("chi-30_m", 0)})[2] == "FAIL"                # start state fails -> no level passes
    assert r4.verdict([])[2] == "INCOMPLETE"


def test_step4_resume_skips_done_states(tmp_path):
    import run_step4 as r4
    model = json.load(open(r4.MODEL))
    res = str(tmp_path / "res.csv")
    class Crash(_FakeRunner):
        def relax(self, rdir, tag, *a):
            if os.path.basename(rdir) == "chi+0_p" and tag.startswith("05_"):
                raise RuntimeError("simulated crash")
            return super().relax(rdir, tag, *a)
    with pytest.raises(RuntimeError):
        r4.run_all(Crash(), model, results=res, work=str(tmp_path / "w"), say=lambda m: None, read=_fake_cell)
    n1 = len(r4.done_rows(res))
    fr = _FakeRunner()
    rows = r4.run_all(fr, model, results=res, work=str(tmp_path / "w"), say=lambda m: None, read=_fake_cell)
    assert len(rows) == 6 * 33 and n1 == 2 * 33 + 5
    assert len({(r["run"], r["idx"]) for r in rows}) == 6 * 33                     # no duplicates
    assert fr.calls[0][1].startswith("05_") and os.path.basename(fr.calls[0][0]) == "chi+0_p"  # resumed where it stopped


def test_step4_dry_run_starts_no_lammps(monkeypatch, capsys):
    import run_step4 as r4
    def boom(*a, **k):
        raise AssertionError("subprocess started in dry run")
    monkeypatch.setattr(subprocess, "run", boom)
    assert r4.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("33 states (31 relaxations + 2 stability)") == 6 and "step-3 entries missing: none" in out


def test_step4_failed_condition_labels():
    import run_step4 as r4
    d0 = dict(dy=0.0, dz=0.0, topcol=1.2)
    assert r4.core_check(dict(dy=0.05, dz=0.0, topcol=1.2), d0)[1:] == (True, "")
    assert r4.core_check(dict(dy=2.019, dz=1.166, topcol=1.2), d0)[1:] == (False, "site jump")
    assert r4.core_check(dict(dy=0.0, dz=0.0, topcol=1.55), d0)[1:] == (False, "split")
    assert r4.core_check(dict(dy=1.346, dz=0.0, topcol=1.8), d0)[1:] == (False, "site jump+split")


def test_step4_review_flag_keeps_snapshot_and_not_verdict(tmp_path):
    import run_step4 as r4
    model = json.load(open(r4.MODEL))
    class Snap(_FakeRunner):
        def relax(self, rdir, tag, *a):
            r = super().relax(rdir, tag, *a); open(r["data"], "w").write("snapshot " + tag); return r
    def read(path):                                         # one atom moves 0.8 A non-affinely in chi+0_p state 07
        c = _fake_cell(path)
        if "chi+0_p" in path and os.path.basename(path).startswith("07_"):
            c["r"] = c["r"].copy(); c["r"][2, 1] += 0.8
        return c
    rows = r4.run_all(Snap(), model, results=str(tmp_path / "res.csv"), work=str(tmp_path / "w"), say=lambda m: None,
                      read=read, snapdir=str(tmp_path / "snap"))
    summ = r4.run_summary(rows)
    assert summ["chi+0_p"]["review"] == "REVIEW" and summ["chi-30_p"]["review"] == ""
    assert summ["chi+0_p"]["max_nonaffine_A"] > 0.5
    flagged = sorted(f"{r['run']}_{os.path.basename(r['data'])}" for r in rows if r.get("review") == "yes")
    assert flagged and all(f.startswith("chi+0_p_") for f in flagged)
    assert sorted(os.listdir(tmp_path / "snap")) == flagged             # one kept snapshot per flagged state
    assert r4.verdict(rows)[:3] == (0.75, 0.75, "PASS")              # the flag does not change s_4
    st0 = next(r for r in rows if r["run"] == "chi-30_p" and r["phase"] == "start")
    assert st0["start_vs_cell_dr_A"] != "" and "n/a" in st0["offaxis_tau0"]
    lad = next(r for r in rows if r["run"] == "chi-30_p" and r["phase"] == "ladder")
    assert "vs floor" in lad["offaxis_tau0"]                           # tau = 0 state: step-3 Run B tau = 0 measurement


def test_step4_gradual_drift_trips_review_cumulative(tmp_path):
    """D-v02-5 (a): 0.06 A per step over 15 steps (ladder + tau ramp) never trips a step-to-step 0.5 A flag, but does from the start."""
    import run_step4 as r4
    model = json.load(open(r4.MODEL))
    rng = np.random.default_rng(0); base = rng.uniform(0, 10, (50, 3))
    class Snap(_FakeRunner):
        def relax(self, rdir, tag, *a):
            r = super().relax(rdir, tag, *a); open(r["data"], "w").write(tag); return r
    def read(path):
        k = int(os.path.basename(path)[:2]) if os.path.basename(path)[:2].isdigit() else 0
        r = base.copy()
        if "chi-30_p" in path:
            r[7, 1] += 0.06 * min(k, 15)                       # one core atom drifts 0.06 A per state up to 15 states
        return dict(Lx=100.0, Lz=100.0, xz=0.0, ids=np.arange(50), r=r)
    rows = r4.run_all(Snap(), model, results=str(tmp_path / "res.csv"), work=str(tmp_path / "w"), say=lambda m: None,
                      read=read, snapdir=str(tmp_path / "snap"))
    run = [r for r in rows if r["run"] == "chi-30_p" and r.get("nonaffine_step_A") not in (None, "")]
    assert max(float(r["nonaffine_step_A"]) for r in run) < 0.1           # step-to-step: never near 0.5 A
    assert max(float(r["nonaffine_cum_A"]) for r in run) > 0.5            # cumulative from the start: trips the flag
    assert r4.run_summary(rows)["chi-30_p"]["review"] == "REVIEW" and r4.run_summary(rows)["chi+0_p"]["review"] == ""


def test_step4_fallback_failure_at_tau_0p3_names_followup(tmp_path):
    """D-v02-5 (b): a top-level (0.75) failure at tau = 0.3 gives INCOMPLETE and names the follow-up at 0.5."""
    import run_step4 as r4
    model = json.load(open(r4.MODEL))
    idx = next(s["idx"] for s in r4.states(0, 1) if s["phase"] == "load" and s["tau"] == 0.3)
    rows = r4.run_all(_FakeRunner({("chi+0_p", idx)}), model, results=str(tmp_path / "res.csv"), work=str(tmp_path / "w"),
                      say=lambda m: None, read=_fake_cell, snapdir=str(tmp_path / "snap"))
    s4, s, v, note = r4.verdict(rows)
    assert (s4, s, v) == (0.5, None, "INCOMPLETE") and "follow-up at 0.5" in note


# ---------------- 3 Nov: negative fx ("--" crash of the first step-4 launch) ----------------
def test_negative_fx_renders_and_parses():
    """(a) fx < 0 (the step-4 crash: sigma_xy correction -1.46e-7 at tau = 0): the new input parses; the v0.1 input did not."""
    cmd = step_cmd(7, os.path.join(ROOT, "tools/in.step_chi"), "log", "P", "in.data", "o.data", "o.pe", -0.0000001463, 74.0301,
                   14.8366, 146.8746, 1.2364, 74.03, 72.27, 200000, "g.lmp", 6.9939, "0", "1e-6", "0.01", 2000000, fz=-0.0089953921)
    v = cmd_vars(cmd)
    assert v["fxb"] == "0.0000001463" and v["fx"] == "-0.0000001463"
    new = _render(os.path.join(ROOT, "tools/in.step_chi"), v)
    assert bad_numbers(new, cmd) == []
    assert any(re.sub(r"\s+", " ", l) == "fix fbot bot addforce 0.0000001463 0.0 0.0089953921" for l in new)
    old = _render(os.path.join(ROOT, "tests/data/in.step_chi_v01"), v)       # the v0.1 template with the same values
    assert any("--0.0000001463" in b for b in bad_numbers(old))


def test_bad_numbers_rejects_malformed():
    for tok in ("--0.1", "+-0.1", "nan", "inf", "-inf", "abc"):
        assert bad_numbers([f"fix f g addforce {tok} 0.0 0.0"]) and bad_numbers([], ["-var", "fx", tok])
    assert bad_numbers(["fix f g addforce -0.1 0.0 +0.2"], ["-var", "fx", "-0.1", "-var", "pot", "--not-a-number-ok"]) == []


def test_step4_render_check_all_198_states():
    """(b) every step-4 input (186 relaxations + 12 stability seeds) renders with all numeric parameters parsing."""
    import run_step4 as r4
    n, problems = r4.render_check(json.load(open(r4.MODEL)))
    assert n == 198 and problems == []


def test_v01_sweep_input_rendering_unchanged():
    """(c) a v0.1 sweep step (chi = +30, tau = 2.0, strain_model.json, fx > 0, sigma_yz = 0): same mpirun command and the same
    evaluated input lines as the v0.1 template; the only difference is the three default declarations (fz, fzb, fxb)."""
    from loading_v02 import model_correction, surface_force
    entry = json.load(open(os.path.join(ROOT, "checks/07_chi_calibration/strain_model.json")))["correction"]["30"]
    LIN = np.array([0.0, 0.0027625, 0.0164349]); tau, chi = 2.0, np.radians(30)
    c = model_correction(entry, tau); e = LIN * tau * np.sin(chi) + c[:3]
    Lx, Lz = 14.8366 * (1 + e[0]), 146.8746 * (1 + e[1]); xz = 1.2364 + e[2] * Lz
    fx = surface_force(tau * np.cos(chi) + c[3], Lx, Lz, 756)
    a = (7, "/R", "coarse_2.000", "prev.data", fx, 74.0301, Lx, Lz, xz, 74.03, 72.27 * (1 + e[1]), 200000, 6.99393, "0", "1e-6", "0.01", 2000000)
    old_cmd = _v01_cmd(*a)
    new_cmd = step_cmd(7, "/R/tools/in.step_chi", "steps/log.coarse_2.000", "/R/potentials/Fe_mm.eam.fs", "prev.data", "steps/coarse_2.000.data",
                       "steps/coarse_2.000.pe", fx, 74.0301, Lx, Lz, xz, 74.03, 72.27 * (1 + e[1]), 200000, "groups.lmp", 6.99393,
                       "0", "1e-6", "0.01", 2000000)
    assert new_cmd == old_cmd                                              # byte-identical command
    v = cmd_vars(new_cmd)
    old = "\n".join(_render(os.path.join(ROOT, "tests/data/in.step_chi_v01"), v))
    new = "\n".join(l for l in _render(os.path.join(ROOT, "tools/in.step_chi"), v)
                    if not re.match(r"variable\s+(fz|fzb|fxb)\s+index", l))
    assert new == old                                                      # byte-identical evaluated input (declarations aside)


# ---------------- 3 Nov: stability-seed argparse crash ("--model_key -30,+0.75") ----------------
def test_stability_negative_model_key_parses():
    import run_step4 as r4
    from stability_args import make_parser
    a = make_parser().parse_args(r4.stab_argv("x.data", 0.6, -30, "t", 1, 7, "-30,-0.75", -0.75))
    assert a.model_key == "-30,-0.75" and a.syz == -0.75 and a.chi == -30 and a.seed == 1
    with pytest.raises(SystemExit):                                   # the old form: what crashed the helper on 3 Nov
        make_parser().parse_args(["x.data", "0.6", "-30", "74.0301", "t", "--model_key", "-30,+0.75"])


def test_stability_helper_nonzero_exit_raises(monkeypatch):
    import run_step4 as r4
    runner = r4.LammpsRunner(7)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 2, "", "error: argument --model_key"))
    with pytest.raises(RuntimeError, match="crashed.*rc 2.*model_key"):
        runner.stability(".", "t", "x.data", "-30,+0.75", -30, 0.6, 0.75, 1)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "no result line", ""))
    with pytest.raises(RuntimeError, match="no 'returned='"):
        runner.stability(".", "t", "x.data", "0,+0.75", 0, 0.6, 0.75, 1)


def test_step4_stability_argv_parse_check_all_12():
    import run_step4 as r4
    assert r4.stab_parse_check() == (12, [])


# ---------------- steps 5 and 6 (V02_PLAN 5.4 rules 3-4, 8.11 D-v02-6), fake runners ----------------
class _RampFake:
    """Moves (glide jump, 2.331 A) at tau >= thr[(chi, syz)]; optional anomaly at given (chi, syz, tau); pre-ramp move option."""
    def __init__(self, thr, anomaly=(), pre_move=(), stab_ok=True):
        self.thr, self.anomaly, self.pre_move, self.stab_ok, self.cell, self.calls = thr, set(anomaly), set(pre_move), stab_ok, "cell", []

    def setup(self, rdir):
        os.makedirs(rdir, exist_ok=True)

    def relax(self, rdir, tag, datain, entry, chi, tau, syz):
        self.calls.append(tag)
        run_syz = float(os.path.basename(rdir).split("syz")[1])
        moved = tau >= self.thr[(chi, run_syz)] - 1e-9 or (tag.split("_")[1] == "preyz" and (chi, run_syz) in self.pre_move)
        an = (chi, run_syz, round(tau, 2)) in self.anomaly
        dz = 2.3313 if moved else (0.5 if an else 0.0)
        return dict(data=os.path.join(rdir, tag), pe=-1.0, fmax=1e-8, fnorm=1e-7, stop="force tolerance",
                    dd=dict(dy=0.0, dz=dz, orient="down", topcol=1.2, col=(0.0, 0.0)))

    def stability(self, rdir, tag, data, key, chi, tau, syz, seed):
        self.calls.append(tag); return self.stab_ok, "returned=" + ("yes" if self.stab_ok else "no")


def _ramp(tmp_path, fake, chi=0, syz=0.0, res="r.csv"):
    import ramp_v02 as rp, run_step4 as r4
    return rp.run_ramp(chi, syz, fake, json.load(open(r4.MODEL)), str(tmp_path / res), str(tmp_path / "w"), say=lambda m: None)


def test_ramp_schedule_bracket_and_stability(tmp_path):
    fake = _RampFake({(0, 0.0): 1.315})
    s = _ramp(tmp_path, fake)
    assert (s["last_unmoved"], s["first_moved"], s["event"]) == (1.31, 1.32, "glide_jump") and abs(s["mid"] - 1.315) < 1e-9
    phases = [c.split("_")[1] for c in fake.calls if not c.startswith("chi")]
    assert phases.count("coarse") == 27 and phases.count("fine") == 2              # coarse 0.05..1.35 (1.35 moved), fine 1.31, 1.32
    assert [c for c in fake.calls if "stab" in c] == ["chi+0_syz+0.00_stab1", "chi+0_syz+0.00_stab2"] and s["rule3_pass"]
    fake2 = _RampFake({(30, -0.75): 2.0}); s2 = _ramp(tmp_path, fake2, 30, -0.75, "r2.csv")
    assert [c.split("_")[1] for c in fake2.calls[:4]] == ["start", "preyz", "preyz", "preyz"]
    assert (s2["last_unmoved"], s2["first_moved"]) == (1.99, 2.0) or (s2["last_unmoved"], s2["first_moved"]) == (1.95, 2.0)


def test_ramp_resume_replays_without_rerun(tmp_path):
    import ramp_v02 as rp, run_step4 as r4
    class Crash(_RampFake):
        def relax(self, rdir, tag, *a):
            if "coarse_syz+0.00_tau1.00" in tag:
                raise RuntimeError("simulated crash")
            return super().relax(rdir, tag, *a)
    with pytest.raises(RuntimeError):
        _ramp(tmp_path, Crash({(0, 0.0): 1.315}))
    n1 = len(rp.done_rows(str(tmp_path / "r.csv")))
    fake = _RampFake({(0, 0.0): 1.315}); s = _ramp(tmp_path, fake)
    rows = rp.done_rows(str(tmp_path / "r.csv"))
    assert n1 == 1 + 19 and len(rows) == 1 + 27 + 2 + 2 and len({(r["phase"], r["tau"]) for r in rows}) == len(rows)
    assert fake.calls[0].endswith("coarse_syz+0.00_tau1.00") and abs(s["mid"] - 1.315) < 1e-9


def test_ramp_rule3_failures(tmp_path):
    assert not _ramp(tmp_path, _RampFake({(0, 0.0): 1.315}, anomaly={(0, 0.0, 0.8)}), res="a.csv")["checks"]["first_event_ok"]
    s = _ramp(tmp_path, _RampFake({(0, 0.75): 9.9}, pre_move={(0, 0.75)}), 0, 0.75, "b.csv")
    assert not s["checks"]["no_preramp_move"] and not s["rule3_pass"]
    s = _ramp(tmp_path, _RampFake({(0, 0.0): 2.99}), res="c.csv")
    assert s["checks"]["bracket"] and not s["checks"]["below_top"]                # first move at 2.99 > 2.95
    s = _ramp(tmp_path, _RampFake({(0, 0.0): 9.9}), res="d.csv")
    assert not s["checks"]["bracket"] and not s["rule3_pass"]                     # no move up to 3.0 GPa
    assert not _ramp(tmp_path, _RampFake({(0, 0.0): 1.315}, stab_ok=False), res="e.csv")["checks"]["stability"]


def _sum(mid, ok=True):
    return dict(mid=mid, rule3_pass=ok, run="r", checks={})


def test_step5_verdict_logic():
    import run_step5 as s5
    A = {-30: _sum(1.165), 0: _sum(1.315), 30: _sum(2.045)}
    B = {(c, s): _sum(A[c]["mid"] + 0.05) for c in (-30, 0, 30) for s in (0.75, -0.75)}
    assert s5.verdict(A, B, 0.75)[:2] == ("PASS", 0)
    assert s5.verdict({**A, 0: _sum(1.34)}, B, 0.75)[:2] == ("STOP", 2)                     # phase A off by 0.025 > 0.02
    v = s5.verdict(A, {**B, (30, -0.75): _sum(2.1, False)}, 0.75)
    assert v[:2] == ("INCOMPLETE", 1) and "repeat ALL six at s = 0.50" in v[2] and "step-4 full-sequence follow-up" in v[2]
    assert s5.verdict(A, {**B, (0, 0.25): _sum(1.3, False)}, 0.25)[:2] == ("STOP", 2)       # failure at 0.25 -> stop
    flat = {k: _sum(A[k[0]]["mid"] + 0.01) for k in B}
    assert s5.verdict(A, flat, 0.75)[:2] == ("STOP", 3)                                     # rule 4 feasibility stop


def _run(mid, ev="glide_jump", seeds=True):
    return dict(mid=mid, first_event=ev, checks=dict(stability=seeds))


def test_step6_verdict_3pct():
    import run_step6 as s6
    one = {(0, 0.0): _run(1.315), (0, 0.75): _run(1.40), (0, -0.75): _run(1.25, "split")}
    ok2 = {(0, 0.0): _run(1.30), (0, 0.75): _run(1.42), (0, -0.75): _run(1.26, "split")}
    assert s6.verdict(ok2, one)[:2] == ("PASS", 0)
    v = s6.verdict({**ok2, (0, 0.75): _run(1.46)}, one)                                        # 4.3 % at +0.75
    assert v[:2] == ("STOP", 2) and [r["pass_"] for r in v[2]] == [True, False, True]
    assert s6.verdict({**ok2, (0, 0.0): _run(None, None)}, one)[1] == 2                        # no bracket -> stop


def test_step6_event_class_differs_stops():
    """D-v02-7 item 5: a 2x run whose first event differs from its 1x run's is not a size test -> STOP."""
    import run_step6 as s6
    one = {(0, 0.0): _run(1.315), (0, 0.75): _run(1.40), (0, -0.75): _run(1.25, "split")}
    two = {(0, 0.0): _run(1.315), (0, 0.75): _run(1.40, "split"), (0, -0.75): _run(1.25, "split")}
    v = s6.verdict(two, one)
    assert v[:2] == ("STOP", 2) and [r["same_event"] for r in v[2]] == [True, False, True]


def test_step6_seed_fails_stops():
    import run_step6 as s6
    one = {(0, 0.0): _run(1.315), (0, 0.75): _run(1.40), (0, -0.75): _run(1.25)}
    two = {(0, 0.0): _run(1.315, seeds=False), (0, 0.75): _run(1.40), (0, -0.75): _run(1.25)}
    assert s6.verdict(two, one)[:2] == ("STOP", 2)


def test_step5_phase_a_rule3_anomaly_or_seed_stops(tmp_path):
    """D-v02-7 item 1: phase A also needs rule 3 -> an anomaly first event, or a failed seed, gives STOP (exit 2)."""
    import run_step5 as s5
    A = {-30: _ramp(tmp_path, _RampFake({(-30, 0.0): 1.165}), -30, 0.0, "a1.csv"),
         0: _ramp(tmp_path, _RampFake({(0, 0.0): 1.315}, anomaly={(0, 0.0, 1.0)}), 0, 0.0, "a2.csv"),
         30: _ramp(tmp_path, _RampFake({(30, 0.0): 2.045}), 30, 0.0, "a3.csv")}
    assert abs(A[0]["mid"] - 1.315) < 1e-9                                                     # the midpoint itself is fine
    v = s5.verdict(A, None, 0.75)
    assert v[:2] == ("STOP", 2) and "rule 3" in v[2] and "first_event_ok" in v[2]
    A[0] = _ramp(tmp_path, _RampFake({(0, 0.0): 1.315}, stab_ok=False), 0, 0.0, "a4.csv")
    v = s5.verdict(A, None, 0.75)
    assert v[:2] == ("STOP", 2) and "stability" in v[2]


def test_steps_5_6_pre_run_checks():
    import ramp_v02 as rp, run_step4 as r4, run_step5 as s5, run_step6 as s6
    model = json.load(open(r4.MODEL))
    A, B = s5.specs(0.75)
    n, na, p = rp.pre_run_checks(A + B, model)
    assert p == [] and n == 9 * 301 + 6 * 3 and na == 2 * (9 * 300 + 9)
    n, na, p = rp.pre_run_checks([(c, s, rp.CELL_2X) for c, s in s6.RUNS], model)
    assert p == [] and n == 3 * 301 + 2 * 3
