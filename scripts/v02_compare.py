"""v0.2 comparison (V02_PLAN 8 Q8/Q9, 8.12 items 1, 5-9, D-v02-9 item 2, D-v02-11): active learning vs random, space-filling
and the nested regular grid, each picking from the pool P and scored on the valid T nodes.

The scoring is the pool-study code, imported (not copied) from scripts/v02_pool_study.py: run_order (random, space-filling),
run_al, run_grid, fit_eval, the EventGPD model, scoring from N_INIT = 3 picks, grid GP random_state 0. The designs are those
of src/surrogate/designs.py. Two module globals of the pool study are set for the length of a run and restored afterwards:
  XT     -> the scaled valid T nodes (the pool study's synthetic T is the same 7 x 7 layout);
  TARGET -> -inf in the default full-trajectory mode, so each run continues to the end of P (needed for the "stayed <= 0.02"
            and Q9 curves, D-v02-11 item 2). The pick order does not depend on TARGET (it only stops the loop), so the first
            crossing in the full trajectory is exactly the pool study's stopping pick. --stop-at-crossing keeps TARGET = 0.02
            (the pool study unchanged; used by the reproduction test, D-v02-11 item 4b).
  fit_eval is wrapped (the original is called) to record, at every pick, the RMSE and the per-node errors and events.
  grid_designs is wrapped to apply the D-v02-9 item 2 substitution of invalid grid nodes.

Definitions (V02_PLAN):
  picks needed = first pick with RMSE <= 0.02 GPa on the valid T nodes; never reached = |P| + 1 (377 for the 376-point P).
  Q8: AL wins seed k only if AL_k < min(random_k, space-filling_k, grid); a saving is claimed only if AL wins >= 15 of 20.
  Ties (D-v02-11 item 1): "same pick" (both reached at the same pick count) or "neither reached" (both |P| + 1).
  Invalid points (8.12 item 8) are excluded for all methods (D-v02-13 b): random / space-filling build their order exactly
  as on the full P (the registered seeded order) and an invalid point is skipped when it comes up, without counting as a
  pick; AL takes its seeded start from the full-P random order (invalid entries skipped) and predicts over the full-P
  candidate set, and an invalid point whose spread ranks above the chosen valid point is removed from the candidates
  (skipped, not a pick). With no invalid point every method is the pool-study code path unchanged. Grid: an invalid node is
  replaced by the nearest valid pool point in (chi / 30, sigma_yz / s), ties to the lower pool index (D-v02-9 item 2); a
  replacement already in that design is not added twice (D-v02-13 c).
  Stop for Ansh if > 2 of 49 T or > 18 of 376 P points are invalid: no comparison is computed.
  P1' / P2 (8.4, descriptive on T, 8.12 item 7): P1' at a run's first-crossing model (its last pick if never reached):
  <= 1 T node with a wrong event AND |error| > 0.02 GPa; P2: after pick 30 no RMSE rise > 0.05 GPa between consecutive
  10-pick checks (grid: between consecutive designs with >= 30 points).
  8.8 flag (8.12 item 9): T nodes with tau_P < 1.0 GPa listed; RMSE at each run's first-crossing model with and without them.

Modes:
  --synthetic SURFACE (weak / medium / strong; SYNTHETIC, src/surrogate/synthetic2d.py), the default. Never reads data/sealed.
           Verdict fields go under "synthetic_tally_code_test_not_a_result" (D-v02-14 item 5).
  --final: real P (data/production/P_results.csv) and sealed T (data/sealed/T_results.csv), behind the seal guard
           (D-v02-11 item 3; full 40-character hash on the dedicated V02_PLAN line, D-v02-14 item 4). Run once.
  --reproduce: replay every run recorded in data/synthetic_v02/pool_study_runs.json through the imported pool-study code
           with its own settings (ps.task: random cap N_CAP at M = 2000), first-crossing mode, and compare every recorded
           field exactly; stop at the first difference (D-v02-14 items 1-2). --reproduce-dry-run prints the counts only.
  --workers N: runs in parallel over (method, seed) / recorded runs; each run single-threaded with its own seeds (D-v02-13 d).
  In first-crossing mode (--stop-at-crossing, --reproduce) "stayed <= 0.02" and the largest RMSE after the crossing are None
  (not observed; D-v02-14 item 3).
Usage (ml env): python scripts/v02_compare.py --synthetic medium --seeds 130 [--workers 7]
                python scripts/v02_compare.py --reproduce [--reproduce-dry-run] [--workers 7]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, contextlib, csv, hashlib, json, math, multiprocessing, re, subprocess, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import v02_pool_study as ps                                         # scoring code of record (reused, not copied)
from surrogate.designs import grid_T, make_pool, scaled, random_order, spacefill_order, grid_designs
from surrogate.synthetic2d import truth, ALPHA0

SEEDS = list(range(130, 150))                                        # 8.12 item 5
P_SIZE, S_GPA, TARGET = 376, 0.75, ps.TARGET
CLAIM_WINS, MAX_INVALID = 15, dict(T=2, P=18)                        # Q8; 8.12 item 8
LOW_TAU = 1.0                                                        # 8.8 flag, GPa
METHODS = ("al", "random", "spacefill", "grid")
SEALED_T = os.path.join(ROOT, "data", "sealed", "T_results.csv")
P_RESULTS = os.path.join(ROOT, "data", "production", "P_results.csv")
T_SHA256 = "5bec6137421be78a3222783df8a54490b1b72631cdcb8b63ab434887148813bc"     # D-v02-11 item 3
PLAN = os.path.join(ROOT, "V02_PLAN.md")
FINAL_DIR = os.path.join(ROOT, "data", "v02_compare")
FINAL_OUT = os.path.join(FINAL_DIR, "final.json")
SYN_DIR = os.path.join(ROOT, "data", "synthetic_v02", "compare")
THIS = os.path.relpath(os.path.abspath(__file__), ROOT)
POOL_RUNS = os.path.join(ROOT, "data", "synthetic_v02", "pool_study_runs.json")          # D-v02-14 item 1 (ext file out of scope)
HASH_LINE = re.compile(r"^v02_compare\.py commit: ([0-9a-f]{40})$", re.M)              # D-v02-14 item 4
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
SYN_VERDICT = "synthetic_tally_code_test_not_a_result"                                   # D-v02-14 item 5


# ---------------------------------------------------------------- pure helpers (unit-tested)
def picks_needed(curve, n_total, target=TARGET):
    """curve: [(picks, rmse), ...] in pick order. First crossing of rmse <= target; never reached -> n_total + 1."""
    for n, r in curve:
        if r <= target:
            return n
    return n_total + 1


def after_crossing(curve, n_total, target=TARGET):
    """(stayed <= target for all later picks, largest RMSE after the first crossing); (None, None) if never reached."""
    k = picks_needed(curve, n_total, target)
    if k > n_total:
        return None, None
    later = [r for n, r in curve if n > k]
    return (all(r <= target for r in later), max(later) if later else None)


def compare_seed(al, others, n_total):
    """Q8 per seed: 'win' only if al < min(others); a tie is 'tie: same pick' or 'tie: neither reached'; else 'loss'."""
    best = min(others)
    if al < best:
        return "win"
    if al == best:
        return "tie: neither reached" if al > n_total else "tie: same pick"
    return "loss"


def tally(outcomes):
    n = len(outcomes)
    out = {}
    for key in ("win", "tie: same pick", "tie: neither reached", "loss"):
        c = sum(o == key for o in outcomes)
        out[key] = dict(count=c, percent=round(100.0 * c / n, 1) if n else None)
    out["tie (all)"] = dict(count=out["tie: same pick"]["count"] + out["tie: neither reached"]["count"],
                            percent=None if not n else round(100.0 * (out["tie: same pick"]["count"]
                                                                      + out["tie: neither reached"]["count"]) / n, 1))
    return out


def nearest_valid_map(P, valid):
    """D-v02-9 item 2: invalid pool index -> nearest valid pool index in scaled x (chi / 30, sigma / s); ties -> lower index."""
    X = scaled(P)
    ok = np.where(valid)[0]
    m = {}
    for i in np.where(~np.asarray(valid))[0]:
        d = np.round(np.linalg.norm(X[ok] - X[i], axis=1), 12)
        m[int(i)] = int(ok[np.lexsort((ok, d))[0]])                   # smallest distance, then lowest index
    return m


def substitute(design, sub):
    """Replace invalid nodes (keys of sub); a replacement already in the design is not added twice."""
    out = []
    for i in design:
        j = sub.get(int(i), int(i))
        if j not in out:
            out.append(j)
    return out


def p2_pass(curve, start=30, step=10, rise=0.05, regular=True):
    """8.4 P2: after pick `start`, no RMSE rise > `rise` between consecutive checks (every `step` picks; grid: every design)."""
    d = dict(curve)
    if regular:
        pts = [d[n] for n in range(start, max(d) + 1, step) if n in d]
    else:
        pts = [r for n, r in curve if n >= start]
    return all(b - a <= rise for a, b in zip(pts, pts[1:]))


def model_keys_all():
    """Every (chi, sigma_yz) pair the production runs use: T and P targets, their pre-ramp levels and the sigma_yz = 0 starts."""
    sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "checks", "11_core_gate"))
    import make_jobs_v02 as mj
    import run_step4 as r4
    tj, pj = mj.build()
    pairs = set()
    for j in tj + pj:
        pairs.add((float(j["chi"]), float(j["syz"])))
        pairs.add((float(j["chi"]), 0.0))
        pairs.update((float(j["chi"]), float(v)) for v in j["levels"])
    return {p: r4.model_key(*p) for p in pairs}


# ---------------------------------------------------------------- seal guard (D-v02-11 item 3)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_pytest(root):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode == 0 and " failed" not in tail and " error" not in tail, tail


def seal_guard(root=ROOT, plan=PLAN, sealed=SEALED_T, final_out=FINAL_OUT, script=THIS, sha=T_SHA256, pytest_fn=run_pytest):
    """Returns the list of failed conditions (empty = may run). Prints and reveals no T values; reads the sealed file only
    to hash it. Cheap checks first; pytest last and only if the others pass."""
    fails = []
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True)
    if st.returncode != 0 or st.stdout.strip():
        fails.append("modified or staged tracked files (git status --porcelain --untracked-files=no is not empty)")
    h = subprocess.run(["git", "log", "-1", "--format=%H", "--", script], cwd=root, capture_output=True, text=True).stdout.strip()
    txt = open(plan).read() if os.path.exists(plan) else ""
    lines = HASH_LINE.findall(txt)
    if not h or lines != [h]:                                        # exactly one dedicated line, full 40-character hash
        fails.append(f"commit hash of {script} not recorded in {os.path.basename(plan)} "
                     f"(one line 'v02_compare.py commit: <full hash>' required)")
    if not os.path.exists(sealed) or sha256(sealed) != sha:
        fails.append("sha256 of the sealed T results does not match the recorded value")
    if os.path.exists(final_out):
        fails.append(f"final output already exists ({os.path.relpath(final_out, root)}): the comparison is run once")
    if not fails:
        ok, tail = pytest_fn(root)
        if not ok:
            fails.append(f"pytest not 0 failed ({tail})")
    return fails


# ---------------------------------------------------------------- running the methods with the pool-study code
@contextlib.contextmanager
def patched(XTv, full, sub):
    """Set the pool-study globals for one run (see module docstring) and record every fit; always restored."""
    saved = dict(XT=ps.XT, TARGET=ps.TARGET, fit_eval=ps.fit_eval, grid_designs=ps.grid_designs)
    rec = []
    orig_fit = saved["fit_eval"]

    def fit_rec(P, y, ev, idx, yT, evT, seed):
        m, r, pev = orig_fit(P, y, ev, idx, yT, evT, seed)
        mu = m.predict(ps.XT)[0]                                     # same deterministic mean that fit_eval scored
        rec.append(dict(n=len(idx), rmse=r, idx=[int(i) for i in idx], err=(mu - yT).tolist(), pev=list(pev)))
        return m, r, pev

    ps.XT, ps.fit_eval = XTv, fit_rec
    ps.TARGET = -math.inf if full else TARGET
    ps.grid_designs = lambda nc, ns: [substitute(d, sub) for d in saved["grid_designs"](nc, ns)]
    try:
        yield rec
    finally:
        for k, v in saved.items():
            setattr(ps, k, v)


def skip_invalid(order, valid):
    """D-v02-13 b: the registered order on the full P with the invalid points removed, sequence kept."""
    return [int(i) for i in order if valid[int(i)]]


def run_al_skip(P, y, ev, yT, evT, seed, valid):
    """ps.run_al on the full P with invalid points skipped (D-v02-13 b). Seeded start: the first N_INIT valid entries of
    random_order(|P|, seed); spread predicted over the full-P free set with the same draw generator; the pick is the valid point
    of largest spread, and any invalid point ranked above it is removed from the candidates (skipped, not a pick).
    With every point valid this is ps.run_al step for step (tested)."""
    valid = np.asarray(valid, bool)
    idx, hist = skip_invalid(random_order(len(P), seed), valid)[:ps.N_INIT], []
    rng = np.random.default_rng(seed + 1000)
    XP = scaled(P)
    gone = []                                                        # invalid points already skipped
    while True:
        m, r, pev = ps.fit_eval(P, y, ev, idx, yT, evT, seed); hist.append(r)
        if r <= ps.TARGET:
            return len(idx), hist
        free = np.setdiff1d(np.arange(len(P)), idx + gone)
        if not valid[free].any():
            return int(valid.sum()) + 1, hist
        _, s, _ = m.predict(XP[free], rng=rng)
        for k in np.argsort(-s, kind="stable"):                      # argmax first (ties: lowest index, as np.argmax)
            i = int(free[k])
            if valid[i]:
                idx = idx + [i]
                break
            gone.append(i)


def run_method(method, seed, D, full=True):
    """One method and seed on dataset D (see dataset()). Returns the record of the run."""
    v = D["valid"]
    P, y, ev = D["P"], D["y"], D["ev"]
    with patched(D["XT"], full, D["sub"]) as rec:
        if method in ("random", "spacefill"):
            order = random_order(len(P), seed) if method == "random" else spacefill_order(P, seed)
            order = skip_invalid(order, v)
            ps.run_order(order, P, y, ev, D["yT"], D["evT"], seed, len(order))
        elif method == "al":
            if v.all():
                ps.run_al(P, y, ev, D["yT"], D["evT"], seed)         # the pool-study code path, unchanged
            else:
                run_al_skip(P, y, ev, D["yT"], D["evT"], seed, v)
        else:
            ps.run_grid(D["P"], D["nc"], D["ns"], D["y"], D["ev"], D["yT"], D["evT"])
    n_total = D["n_total"]
    curve = [(r["n"], r["rmse"]) for r in rec]
    k = picks_needed(curve, n_total)
    at = next((r for r in rec if r["n"] == k), rec[-1])               # first-crossing model (last pick if never reached)
    err, pev = np.array(at["err"]), np.array(at["pev"])
    wrong = pev != D["evT"]
    low = D["low"]
    stayed, worst = after_crossing(curve, n_total) if full else (None, None)    # D-v02-14 item 3: not observed
    return dict(method=method, seed=seed, picks=k, reached=k <= n_total, stayed_after=stayed, max_rmse_after=worst,
                picked=[int(i) for i in at["idx"]],                  # D-v02-16 (1): ordered picks up to the crossing
                curve=curve, p1prime_points=int((wrong & (np.abs(err) > TARGET)).sum()),
                p1prime_pass=bool((wrong & (np.abs(err) > TARGET)).sum() <= 1),
                p2_pass=p2_pass(curve, regular=method != "grid"), event_wrong=int(wrong.sum()),
                rmse_at=float(np.sqrt(np.mean(err ** 2))),
                rmse_at_without_low=None if low.all() else float(np.sqrt(np.mean(err[~low] ** 2))))


def add_pick_ids(runs, P_ids):
    """D-v02-16 (1), --final: the picked pool indices of every run mapped to P IDs (pool index i = P_ids[i])."""
    for rs in runs.values():
        for r in rs:
            r["picked_ids"] = [P_ids[i] for i in r["picked"]]
    return runs


def dataset(P, nc, ns, y, ev, valid, T, yT, evT, n_total):
    valid = np.asarray(valid, bool)
    return dict(P=P, nc=nc, ns=ns, y=np.asarray(y, float), ev=np.asarray(ev), valid=valid, XT=scaled(T), yT=np.asarray(yT, float),
                evT=np.asarray(evT), n_total=n_total, sub=nearest_valid_map(P, valid), low=np.asarray(yT, float) < LOW_TAU)


def synthetic_dataset(surface, M=P_SIZE):
    P, nc, ns = make_pool(M)
    y, ev = truth(P[:, 0], P[:, 1], surface)
    T = grid_T()
    yT, evT = truth(T[:, 0], T[:, 1], surface)
    return dataset(P, nc, ns, y, ev, np.ones(len(P), bool), T, yT, evT, len(P))


_WORK = {}                                                           # dataset shared with forked workers


def _run_task(t):
    m, s = t
    return run_method(m, s, _WORK["D"], _WORK["full"])


def pmap(fn, tasks, workers, say=print, label="run"):
    """Ordered map over tasks; workers > 1: forked processes (each run single-threaded, its own seeds). Output order = task order."""
    t0, out = time.time(), []
    if workers <= 1:
        for k, t in enumerate(tasks):
            out.append(fn(t)); say(f"{label} {k + 1}/{len(tasks)} done: {time.time() - t0:.0f} s")
        return out
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for k, r in enumerate(pool.imap(fn, tasks)):
            out.append(r); say(f"{label} {k + 1}/{len(tasks)} done: {time.time() - t0:.0f} s")
    return out


def compare(D, seeds, full=True, say=print, workers=1):
    tasks = [("grid", 0)] + [(m, s) for s in seeds for m in ("al", "random", "spacefill")]
    _WORK.update(D=D, full=full)
    try:
        res = pmap(_run_task, tasks, workers, say)
    finally:
        _WORK.clear()
    g = res[0]                                                       # grid: deterministic, one run, compared with every seed
    runs = {m: [r for r in res[1:] if r["method"] == m] for m in ("al", "random", "spacefill")}
    runs["grid"] = [g]
    n_total = D["n_total"]
    outcomes = [compare_seed(a["picks"], [r["picks"], f["picks"], g["picks"]], n_total)
                for a, r, f in zip(runs["al"], runs["random"], runs["spacefill"])]
    t = tally(outcomes)
    curves = {}
    for m in METHODS:
        by_n = {}
        for r in runs[m]:
            for n, x in r["curve"]:
                by_n.setdefault(n, []).append(x)
        curves[m] = [(n, float(np.median(v)), float(min(v)), float(max(v)), len(v)) for n, v in sorted(by_n.items())]
    per_seed = [dict(seed=s, al=a["picks"], random=r["picks"], spacefill=f["picks"], grid=g["picks"], outcome=o)
                for s, a, r, f, o in zip(seeds, runs["al"], runs["random"], runs["spacefill"], outcomes)]
    summ = {m: dict(median_picks=float(np.median([r["picks"] for r in runs[m]])),
                    not_reached=sum(not r["reached"] for r in runs[m]),
                    p1prime_pass=sum(r["p1prime_pass"] for r in runs[m]), p2_pass=sum(r["p2_pass"] for r in runs[m]),
                    runs=len(runs[m])) for m in METHODS}
    return dict(n_total=n_total, never_reached_value=n_total + 1, seeds=list(seeds), full_trajectory=full,
                per_seed=per_seed, tally=t, q8=dict(al_wins=t["win"]["count"], needed=CLAIM_WINS, of=len(seeds),
                                                    saving_claimed=t["win"]["count"] >= CLAIM_WINS),
                summary=summ, curves=curves,
                runs=runs,
                grid_substitutions=dict(invalid_nodes_replaced=len([i for i in D["sub"] if i < D["nc"] * D["ns"]]),
                                        map={str(k): v for k, v in D["sub"].items() if k < D["nc"] * D["ns"]}),
                low_tau_nodes=[int(i) for i in np.where(D["low"])[0]])


def label_synthetic(res):
    """D-v02-14 item 5: synthetic verdict fields (per-seed outcome, tally, Q8 claim) moved under SYN_VERDICT."""
    res = dict(res)
    res[SYN_VERDICT] = dict(tally=res.pop("tally"), q8=res.pop("q8"),
                            outcome_per_seed=[dict(seed=r["seed"], outcome=r["outcome"]) for r in res["per_seed"]])
    res["per_seed"] = [{k: v for k, v in r.items() if k != "outcome"} for r in res["per_seed"]]
    return res


# ---------------------------------------------------------------- replay of the pool study (D-v02-14 items 1-2)
def threads_ok(env=None):
    env = os.environ if env is None else env
    return [v for v in THREAD_VARS if env.get(v) != "1"]


def load_records(path=POOL_RUNS):
    return json.load(open(path))


def _replay_task(rec):
    """One recorded run through ps.task (the pool study's own settings, incl. N_CAP), first-crossing mode, with the
    v02_compare recorder attached; returns ps.task's record with v02_compare's own first-crossing count added."""
    with patched(ps.XT, False, {}) as fits:
        out = ps.task((rec["version"], rec["M"], rec["method"], rec["seed"]))
    curve = [(f["n"], f["rmse"]) for f in fits]
    k = picks_needed(curve, out["pool"])
    out["_vc_picks"] = k if k <= max(n for n, _ in curve) else None   # None: not reached within the fitted picks
    return json.loads(json.dumps(out))                               # same float round trip as the recorded JSON


def replay_fields(records):
    """(fields compared, recorded fields first-crossing mode cannot reproduce)."""
    keys = sorted({k for r in records for k in r})
    return [k for k in keys if k != "seconds"], ["seconds (wall-clock time)"]


def replay_estimate(records, workers):
    """Seconds: recorded run times; workers > 1 by longest-first assignment to the least-loaded worker."""
    load = [0.0] * max(1, workers)
    for t in sorted((r.get("seconds", 0.0) for r in records), reverse=True):
        load[load.index(min(load))] += t
    return max(load)


def replay_dry(records, say=print):
    fields, missing = replay_fields(records)
    confs = {(r["version"], r["M"], r["method"]) for r in records}
    say(f"replay: {len(records)} recorded runs, {len(confs)} configurations (surface, M, method), from {os.path.relpath(POOL_RUNS, ROOT)}")
    say(f"fields compared exactly: {', '.join(fields)}; plus v02_compare's own first-crossing count = picks")
    say(f"recorded fields first-crossing mode cannot produce: {', '.join(missing)}")
    say(f"estimate from the recorded run times: 1 worker {replay_estimate(records, 1) / 3600:.1f} h, "
        f"7 workers {replay_estimate(records, 7) / 3600:.1f} h")


def replay_check(records, outs):
    """First difference as (run label, field, recorded, replayed), or None if every compared field matches."""
    fields, _ = replay_fields(records)
    for rec, out in zip(records, outs):
        lab = f"{rec['version']} M={rec['M']} {rec['method']} seed {rec['seed']}"
        for f in fields:
            if f not in out or out[f] != rec.get(f):
                return lab, f, rec.get(f), out.get(f, "<missing>")
        if out["_vc_picks"] is not None and out["_vc_picks"] != rec["picks"]:
            return lab, "picks (v02_compare first crossing)", rec["picks"], out["_vc_picks"]
    return None


def reproduce(records, workers=1, say=print, task=None):
    bad = threads_ok()
    if bad:
        raise SystemExit(f"--reproduce refused (D-v02-14 item 2): threads not 1 for {', '.join(bad)}")
    replay_dry(records, say)
    outs = pmap(task or _replay_task, records, workers, say, label="replayed run")
    diff = replay_check(records, outs)
    if diff:
        raise SystemExit(f"REPRODUCTION STOP: {diff[0]}: field '{diff[1]}' recorded {diff[2]!r}, replayed {diff[3]!r}; "
                         "nothing adjusted")
    say(f"REPRODUCTION OK: {len(records)}/{len(records)} runs, every compared field identical")
    return outs


def write_outputs(res, stem, out_dir, fig_dir, title):
    os.makedirs(out_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
    with open(os.path.join(out_dir, stem + "_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "picks", "rmse_median", "rmse_min", "rmse_max", "n_runs"])
        for m, rows in res["curves"].items():
            w.writerows([[m, n, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", k] for n, a, b, c, k in rows])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, rows in res["curves"].items():
        n = [r[0] for r in rows]
        ax.plot(n, [r[1] for r in rows], label=m, marker="o" if m == "grid" else None, ms=3)
        if m != "grid":
            ax.fill_between(n, [r[2] for r in rows], [r[3] for r in rows], alpha=0.15)
    ax.axhline(TARGET, color="0.5", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("picks from P"); ax.set_ylabel("RMSE on T (GPa); median, band = range over seeds")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, stem + "_rmse_curves.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------- final mode (real data)
def read_real(t_path, p_path):
    """Builds the real dataset; returns (dataset or None, counts dict). Counts only; no values are printed here."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_jobs_v02 as mj
    tj, pj = mj.build()
    rows = {r["id"]: r for path in (t_path, p_path) for r in csv.DictReader(open(path))}
    lab = {"glide_jump": "glide", "split": "split"}

    def take(jobs):
        y, ev, ok, reasons = [], [], [], {}
        for j in sorted(jobs, key=lambda j: j["id"]):
            r = rows.get(j["id"])
            if r is None:
                raise SystemExit(f"refused: {j['id']} has no result row (set incomplete)")
            if abs(float(r["chi"]) - j["chi"]) > 1e-9 or abs(float(r["syz"]) - j["syz"]) > 1e-9:
                raise SystemExit(f"refused: {j['id']} result row is not at its job's (chi, syz)")
            good = r["valid"] == "yes"
            ok.append(good)
            if not good:
                for x in r["reason"].split(";"):
                    k = x.split(":")[0].split(" (")[0]                 # reason class only (no event value)
                    reasons[k] = reasons.get(k, 0) + 1
            y.append(float(r["tau_P_mid"]) if good else np.nan)
            ev.append(lab.get(r["event"], "invalid") if good else "invalid")
        return np.array(y), np.array(ev), np.array(ok), reasons

    yT, evT, okT, rT = take(tj)
    yP, evP, okP, rP = take(pj)
    counts = dict(T=dict(n=len(okT), invalid=int((~okT).sum()), reasons=rT), P=dict(n=len(okP), invalid=int((~okP).sum()), reasons=rP))
    if counts["T"]["invalid"] > MAX_INVALID["T"] or counts["P"]["invalid"] > MAX_INVALID["P"]:
        return None, counts
    P, nc, ns = make_pool(P_SIZE)
    T = grid_T()
    D = dataset(P, nc, ns, yP, evP, okP, T[okT], yT[okT], evT[okT], P_SIZE)
    D["T_ids"] = [j["id"] for j, ok in zip(sorted(tj, key=lambda j: j["id"]), okT) if ok]     # valid T nodes, in D order
    D["P_ids"] = [j["id"] for j in sorted(pj, key=lambda j: j["id"])]                          # pool index -> P ID
    return D, counts


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = a.add_mutually_exclusive_group()
    g.add_argument("--synthetic", choices=sorted(ALPHA0), default="medium")
    g.add_argument("--final", action="store_true")
    a.add_argument("--seeds", default="130-149", help="e.g. 130-149 or 130 or 0,1,2")
    a.add_argument("--pool", type=int, default=P_SIZE, help="synthetic pool size (make_pool(M))")
    a.add_argument("--stop-at-crossing", action="store_true", help="pool-study behaviour: stop each run at the first crossing")
    a.add_argument("--workers", type=int, default=1, help="parallel runs (each single-threaded); default 1")
    a.add_argument("--reproduce", action="store_true", help="replay pool_study_runs.json exactly (D-v02-14)")
    a.add_argument("--reproduce-dry-run", action="store_true", help="with --reproduce: print counts, fields, estimate; run nothing")
    a = a.parse_args(argv)
    if a.reproduce:
        recs = load_records()
        if a.reproduce_dry_run:
            return replay_dry(recs)
        return reproduce(recs, a.workers)
    seeds = [int(x) for p in a.seeds.split(",") for x in
             (range(int(p.split("-")[0]), int(p.split("-")[1]) + 1) if "-" in p else [p])]
    full = not a.stop_at_crossing
    if a.final:
        if seeds != SEEDS or a.pool != P_SIZE or not full:
            raise SystemExit("refused: --final uses seeds 130-149, |P| = 376 and full trajectories only")
        fails = seal_guard()
        if fails:
            raise SystemExit("--final refused (no T values read or printed):\n  - " + "\n  - ".join(fails))
        D, counts = read_real(SEALED_T, P_RESULTS)
        print(f"invalid: T {counts['T']['invalid']} of {counts['T']['n']}, P {counts['P']['invalid']} of {counts['P']['n']}")
        if D is None:
            raise SystemExit("STOP for Ansh (8.12 item 8): invalid count above the limit (> 2 of 49 T or > 18 of 376 P); "
                             "no comparison computed")
        res = compare(D, seeds, full, workers=a.workers)
        add_pick_ids(res["runs"], D["P_ids"])
        res["invalid"] = counts
        res["low_tau_nodes"] = [dict(id=tid, tau_P_mid=float(yv)) for tid, yv, lo in zip(D["T_ids"], D["yT"], D["low"]) if lo]
        res["script_commit"] = subprocess.run(["git", "log", "-1", "--format=%H", "--", THIS], cwd=ROOT,
                                              capture_output=True, text=True).stdout.strip()
        write_outputs(res, "final", FINAL_DIR, os.path.join(ROOT, "figures"), "v0.2 comparison on T (real data), 20 seeds")
        print(f"written {os.path.relpath(FINAL_OUT, ROOT)}")
        return res
    D = synthetic_dataset(a.synthetic, a.pool)
    res = label_synthetic(compare(D, seeds, full, workers=a.workers))
    res["mode"] = f"SYNTHETIC {a.synthetic}, pool {len(D['P'])}"
    stem = f"{a.synthetic}_M{a.pool}_s{seeds[0]}-{seeds[-1]}" + ("_stop" if not full else "")
    write_outputs(res, stem, SYN_DIR, os.path.join(ROOT, "figures", "synthetic"),
                  f"SYNTHETIC ({a.synthetic}) code test, pool {len(D['P'])}, seeds {seeds[0]}-{seeds[-1]}: not a result")
    print(f"SYNTHETIC {a.synthetic} (code test, not a result): per seed "
          f"{[(r['seed'], r['al'], r['random'], r['spacefill'], r['grid']) for r in res['per_seed']]}; "
          f"written {os.path.relpath(os.path.join(SYN_DIR, stem + '.json'), ROOT)}")
    return res


if __name__ == "__main__":
    main()
