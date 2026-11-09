"""Light, hand-checkable tests of scripts/v02_compare.py (D-v02-11 item 4a, seal guard item 3). No sealed file is read."""
import hashlib, json, os, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v02_compare as vc
import v02_pool_study as ps


class FakeModel:
    """Nearest-neighbour stand-in for EventGPD (fast, deterministic); records every training set."""
    fits = []

    def __init__(self, seed=0):
        self.n_split = self.n_glide = 0

    def fit(self, X, y, ev):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.n_glide = len(y)
        FakeModel.fits.append(self.X.copy())
        return self

    def predict(self, X, rng=None):
        d = np.linalg.norm(np.asarray(X)[:, None] - self.X[None], axis=2)
        return self.y[d.argmin(1)], d.min(1), np.array(["glide"] * len(X))


def _recorder(monkeypatch):
    """Wrap ps.fit_eval (before vc.patched wraps it) to record every training set as pool indices."""
    sets, orig = [], ps.fit_eval

    def rec(P, y, ev, idx, yT, evT, seed):
        sets.append([int(i) for i in idx])
        return orig(P, y, ev, idx, yT, evT, seed)
    monkeypatch.setattr(ps, "fit_eval", rec)
    return sets


def _tiny(valid_mask=None, M=40):
    D0 = vc.synthetic_dataset("medium", M)
    valid = np.ones(len(D0["P"]), bool) if valid_mask is None else valid_mask
    return vc.dataset(D0["P"], D0["nc"], D0["ns"], D0["y"], D0["ev"], valid, ps.T, D0["yT"], D0["evT"], len(D0["P"]))


def test_invalid_point_excluded_for_all_methods(monkeypatch):
    """D-v02-13 b: orders built on the full P, invalid points skipped. (i) the order with an invalid point equals the all-valid
    order minus that point, same sequence (random, space-filling; AL: its seeded start); (ii) an invalid point is never
    trained on (all four methods); (iii) the pick count excludes skipped points (training-set size = pick count)."""
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    sets = _recorder(monkeypatch)
    Dall = _tiny()
    start = vc.random_order(len(Dall["P"]), 130)[:3]
    bad = [int(start[1]), 5]                                         # an AL start point, and a regular-part (grid) node
    valid = np.ones(len(Dall["P"]), bool); valid[bad] = False
    D = _tiny(valid)
    for m in vc.METHODS:
        sets.clear(); vc.run_method(m, 130, Dall); full_order = sets[-1]
        sets.clear(); r = vc.run_method(m, 130, D); skip_sets = list(sets)
        assert skip_sets, m
        assert not any(set(bad) & set(x) for x in skip_sets), m                                      # (ii)
        assert all(len(x) == n for x, (n, _) in zip(skip_sets, r["curve"])), m                       # (iii)
        if m in ("random", "spacefill"):
            assert skip_sets[-1] == [i for i in full_order if i not in bad], m                       # (i)
            assert max(n for n, _ in r["curve"]) == valid.sum(), m                                   # (iii)
        if m == "al":
            assert skip_sets[0] == [int(i) for i in vc.random_order(len(D["P"]), 130) if valid[i]][:3]   # (i) start
            assert len(set(skip_sets[-1])) == len(skip_sets[-1]) == valid.sum()                      # every valid point once
    assert ps.MODEL is FakeModel and ps.XT is not D["XT"] and ps.TARGET == 0.02   # globals restored after each run


def test_al_skip_equals_pool_study_al_when_all_valid(monkeypatch):
    """With no invalid point, run_al_skip picks exactly what ps.run_al picks (full trajectory)."""
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    sets = _recorder(monkeypatch)
    D = _tiny()
    with vc.patched(D["XT"], True, {}):
        ps.run_al(D["P"], D["y"], D["ev"], D["yT"], D["evT"], 131)
    a = list(sets); sets.clear()
    with vc.patched(D["XT"], True, {}):
        vc.run_al_skip(D["P"], D["y"], D["ev"], D["yT"], D["evT"], 131, D["valid"])
    assert a and sets == a


def test_stop_mode_stores_none_after_crossing(monkeypatch):
    """D-v02-14 item 3: in first-crossing mode the after-crossing fields are not observed: None, None."""
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    r = vc.run_method("random", 130, _tiny(), full=False)
    assert r["stayed_after"] is None and r["max_rmse_after"] is None


def _fit_cross_at_7(P, y, ev, idx, yT, evT, seed):
    """Stand-in fit_eval: FakeModel, RMSE 0.05 below 7 training points and 0.01 from 7 on (first crossing at 7 picks)."""
    m = FakeModel(seed).fit(vc.scaled(P[idx]), y[idx], ev[idx])
    _, _, pev = m.predict(ps.XT)
    return m, (0.05 if len(idx) < 7 else 0.01), pev


def test_picked_lists_registered_order_invalid_and_grid_substitute(monkeypatch):
    """D-v02-16 (1): (i) random / space-filling / AL lists have length = picks; random and space-filling equal the registered
    order with invalid points removed, up to the crossing; (ii) no invalid point in any list; (iii) grid = the crossing design
    with its substitute; --final IDs map index i to P_ids[i]."""
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    monkeypatch.setattr(ps, "fit_eval", _fit_cross_at_7)
    D0 = _tiny()
    designs = [list(map(int, d)) for d in ps.grid_designs(D0["nc"], D0["ns"])]
    cross = next(d for d in designs if len(d) >= 7)
    order = [int(i) for i in vc.random_order(len(D0["P"]), 130)]
    bad = [cross[1]] + [i for i in order[:5] if i != cross[1]][:1]   # a node of the crossing grid design + an early random pick
    valid = np.ones(len(D0["P"]), bool); valid[bad] = False
    D = _tiny(valid)
    sub = D["sub"]
    runs = {m: vc.run_method(m, 130, D, full=False) for m in vc.METHODS}
    for m, r in runs.items():
        assert not set(bad) & set(r["picked"]), m                                                    # (ii)
    for m in ("random", "spacefill", "al"):
        assert runs[m]["picks"] == 7 and len(runs[m]["picked"]) == 7, m                              # (i)
    assert runs["random"]["picked"] == vc.skip_invalid(order, valid)[:7]                             # (i)
    assert runs["spacefill"]["picked"] == vc.skip_invalid(vc.spacefill_order(D["P"], 130), valid)[:7]
    assert runs["al"]["picked"][:3] == vc.skip_invalid(order, valid)[:3]
    gd = next(vc.substitute(d, sub) for d in designs if len(vc.substitute(d, sub)) >= 7)
    assert runs["grid"]["picked"] == gd and sub[cross[1]] in gd                                     # (iii)
    ids = [f"P{i:03d}" for i in range(len(D["P"]))]
    out = vc.add_pick_ids({"random": [runs["random"]]}, ids)
    assert out["random"][0]["picked_ids"] == [ids[i] for i in runs["random"]["picked"]]


def test_workers_output_identical_and_synthetic_labels(monkeypatch):
    """D-v02-13 d: 2 workers give byte-identical output to 1 worker. D-v02-14 item 5: no top-level verdict field."""
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    D = _tiny()
    one = json.dumps(vc.label_synthetic(vc.compare(D, [130, 131], say=lambda m: None, workers=1)), sort_keys=True).encode()
    two = json.dumps(vc.label_synthetic(vc.compare(D, [130, 131], say=lambda m: None, workers=2)), sort_keys=True).encode()
    assert one == two
    res = json.loads(one)
    assert not {"tally", "q8"} & set(res) and vc.SYN_VERDICT in res
    assert all("outcome" not in r for r in res["per_seed"])


def _fake_record(monkeypatch):
    monkeypatch.setattr(ps, "MODEL", FakeModel)
    out = vc._replay_task(dict(version="medium", M=40, method="random", seed=130))
    rec = {k: v for k, v in out.items() if k != "_vc_picks"}
    rec["seconds"] = 0.5
    return rec


def test_replay_matching_run_passes(monkeypatch):
    rec = _fake_record(monkeypatch)
    for v in vc.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    outs = vc.reproduce([rec], say=lambda m: None)
    assert outs[0]["hist"] == rec["hist"] and outs[0]["picks"] == rec["picks"]


def test_replay_one_field_difference_stops(monkeypatch):
    rec = _fake_record(monkeypatch)
    for v in vc.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    rec["hist"] = [rec["hist"][0] + 1e-15] + rec["hist"][1:]
    with pytest.raises(SystemExit) as e:
        vc.reproduce([rec], say=lambda m: None)
    assert "medium M=40 random seed 130" in str(e.value) and "field 'hist'" in str(e.value)


def test_replay_refuses_unless_threads_are_1(monkeypatch):
    for v in vc.THREAD_VARS:
        monkeypatch.setenv(v, "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with pytest.raises(SystemExit, match="refused.*MKL_NUM_THREADS"):
        vc.reproduce([], say=lambda m: None, task=lambda r: r)


def test_tie_counted_as_tie_not_win():
    assert vc.compare_seed(10, [10, 12, 377], 376) == "tie: same pick"
    assert vc.compare_seed(377, [377, 377, 377], 376) == "tie: neither reached"
    assert vc.compare_seed(9, [10, 12, 377], 376) == "win"
    assert vc.compare_seed(11, [10, 12, 377], 376) == "loss"
    t = vc.tally(["tie: same pick", "tie: neither reached", "win", "loss"])
    assert t["win"]["count"] == 1 and t["tie (all)"]["count"] == 2 and t["loss"]["count"] == 1 and t["win"]["percent"] == 25.0


def test_never_reached_is_377():
    curve = [(n, 0.03) for n in range(3, 377)]
    assert vc.picks_needed(curve, 376) == 377
    assert vc.after_crossing(curve, 376) == (None, None)


def test_first_crossing_at_pick_11():
    curve = [(10, 0.021), (11, 0.019), (12, 0.022), (13, 0.018)]
    assert vc.picks_needed(curve, 376) == 11
    stayed, worst = vc.after_crossing(curve, 376)
    assert stayed is False and worst == 0.022


def test_grid_substitution_lower_index_tie_break():
    # scaled x = (chi / 30, sig): pool points 1 and 2 are both 0.1 from point 0; point 3 is 0.5 away
    P = np.array([[0.0, 0.0], [3.0, 0.0], [-3.0, 0.0], [0.0, 0.5]])
    assert vc.nearest_valid_map(P, np.array([False, True, True, True])) == {0: 1}
    P2 = np.array([[0.0, 0.0], [-3.0, 0.0], [3.0, 0.0]])            # the tie listed the other way round: still lower index
    assert vc.nearest_valid_map(P2, np.array([False, True, True])) == {0: 1}
    assert vc.substitute([0, 2, 3], {0: 1}) == [1, 2, 3]
    assert vc.substitute([0, 1, 3], {0: 1}) == [1, 3]                # the replacement is not added twice


def test_p2_rule():
    ok = [(n, 0.1) for n in range(3, 80)]
    assert vc.p2_pass(ok)
    bad = [(n, 0.1 if n != 50 else 0.16) for n in range(3, 80)]
    assert not vc.p2_pass(bad)
    early = [(n, 0.1 if n != 20 else 0.5) for n in range(3, 80)]     # a rise before pick 30 does not count
    assert vc.p2_pass(early)


def _git(cwd, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def sealed_repo(tmp_path):
    r = tmp_path / "repo"; (r / "scripts").mkdir(parents=True); (r / "data").mkdir()
    (r / "scripts" / "v02_compare.py").write_text("# script\n")
    sealed = r / "data" / "T_results.csv"; sealed.write_text("id,x\nT00,1\n")
    _git(r, "init", "-q"); _git(r, "add", "."); _git(r, "commit", "-q", "-m", "script")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", "scripts/v02_compare.py"], cwd=r, capture_output=True,
                       text=True).stdout.strip()
    (r / "V02_PLAN.md").write_text(f"plan text\nv02_compare.py commit: {h}\n")       # D-v02-14 item 4: dedicated line
    _git(r, "add", "."); _git(r, "commit", "-q", "-m", "plan")
    kw = dict(root=str(r), plan=str(r / "V02_PLAN.md"), sealed=str(sealed), final_out=str(r / "out" / "final.json"),
              script="scripts/v02_compare.py", sha=hashlib.sha256(sealed.read_bytes()).hexdigest(),
              pytest_fn=lambda root: (True, "1 passed"))
    assert vc.seal_guard(**kw) == []                                 # all conditions hold
    return r, kw


def test_seal_guard_refuses_modified_tracked_file(sealed_repo):
    r, kw = sealed_repo
    (r / "scripts" / "v02_compare.py").write_text("# changed\n")
    assert any("modified" in f for f in vc.seal_guard(**kw))


def test_seal_guard_refuses_existing_final_output(sealed_repo):
    r, kw = sealed_repo
    os.makedirs(r / "out"); (r / "out" / "final.json").write_text("{}")
    assert any("already exists" in f for f in vc.seal_guard(**kw))


def test_seal_guard_refuses_wrong_sha256(sealed_repo):
    r, kw = sealed_repo
    assert any("sha256" in f for f in vc.seal_guard(**dict(kw, sha="0" * 64)))


def test_seal_guard_refuses_unrecorded_hash_and_failed_pytest(sealed_repo):
    r, kw = sealed_repo
    (r / "V02_PLAN.md").write_text("no hash here\n"); _git(r, "commit", "-q", "-am", "plan without hash")
    assert any("not recorded" in f for f in vc.seal_guard(**kw))
    _git(r, "revert", "--no-edit", "HEAD")
    assert any("pytest" in f for f in vc.seal_guard(**dict(kw, pytest_fn=lambda root: (False, "1 failed"))))


def test_all_T_and_P_pairs_have_distinct_model_keys():
    keys = vc.model_keys_all()                                       # targets, pre-ramp levels, sigma_yz = 0 starts
    assert len(keys) == 467
    assert len(set(keys.values())) == len(keys)


def test_seal_guard_needs_full_hash_on_dedicated_line(sealed_repo):
    """D-v02-14 item 4: a short prefix elsewhere in V02_PLAN fails; the exact full hash on the dedicated line passes."""
    r, kw = sealed_repo
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", "scripts/v02_compare.py"], cwd=r, capture_output=True,
                       text=True).stdout.strip()
    for txt in (f"script at {h[:7]}, see {h}\n", f"v02_compare.py commit: {h[:12]}\n", f"v02_compare.py commit: {h} (x)\n"):
        (r / "V02_PLAN.md").write_text(txt); _git(r, "commit", "-q", "-am", "plan variant")
        assert any("not recorded" in f for f in vc.seal_guard(**kw)), txt
    (r / "V02_PLAN.md").write_text(f"text {h[:7]}\nv02_compare.py commit: {h}\n"); _git(r, "commit", "-q", "-am", "exact")
    assert vc.seal_guard(**kw) == []
