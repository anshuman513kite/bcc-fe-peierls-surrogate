"""Validation of the Q10-revised model (V02_PLAN section 8.2, 2 Nov 2026): pass criteria P1, P2 (new model) and P3 (old vs new).
SYNTHETIC data only, no LAMMPS. Random selection on make_pool(2000), FRESH seeds 100-109, surfaces weak / medium / strong,
old model (EventGP, section 8 Q10) and new model (EventGPC) on the same random orders: 60 runs.
Run protocol (8.2): continue to 100 picks even after reaching RMSE <= 0.02 GPa, or further (up to 400) while not yet reached.
  picks needed = first pick with RMSE <= 0.02 (401 if never); final pick = the last pick of the run;
  P1: wrong-event predictions on T at the final pick <= 1, in >= 27 of 30 runs (new model);
  P2: no RMSE rise > 0.05 GPa between consecutive 10-pick checks from pick 30 on, in >= 27 of 30 runs (new model);
  P3: median picks needed per surface, old vs new (descriptive).
Output: data/synthetic_v02/validate_gpc.json. Usage (ml env): python v02_validate_gpc.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v02_pool_study import setup, LARGE_POOL, N_CAP, TARGET, N_INIT, XT, OUT, VERSIONS
from surrogate.event_gp import EventGP, EventGPC
from surrogate.designs import random_order, scaled

SEEDS, MIN_PICKS, MAX_RISE = list(range(100, 110)), 100, 0.05
MODELS = {"old": EventGP, "new": EventGPC}


def run(t):
    warnings.filterwarnings("ignore")
    version, seed, model = t
    P, nc, ns, y, ev, yT, evT = setup(version, LARGE_POOL)
    order = random_order(len(P), seed)
    rmse, reached, t0 = {}, None, time.time()
    for n in range(N_INIT, N_CAP + 1):
        m = MODELS[model](seed).fit(scaled(P[order[:n]]), y[order[:n]], ev[order[:n]])
        mu, _, pev = m.predict(XT)
        r = float(np.sqrt(np.mean((mu - yT) ** 2))); rmse[n] = r
        if reached is None and r <= TARGET:
            reached = n
        if n >= MIN_PICKS and reached is not None:
            break
    checks = [k for k in sorted(rmse) if k >= 30 and k % 10 == 0]
    rises = [rmse[b] - rmse[a] for a, b in zip(checks[:-1], checks[1:])]
    wrong = int((pev != evT).sum())
    return dict(version=version, seed=seed, model=model, picks_needed=reached or N_CAP + 1, final_pick=n, wrong_event_final=wrong,
                max_rise=max(rises) if rises else 0.0, rmse_checks={k: rmse[k] for k in checks}, rmse_final=rmse[n],
                p1=wrong <= 1, p2=(max(rises) if rises else 0.0) <= MAX_RISE, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(v, s, mdl) for mdl in ("new", "old") for v in VERSIONS for s in SEEDS]
    with Pool(7) as pool:
        res = pool.map(run, tasks, chunksize=1)
    new = [r for r in res if r["model"] == "new"]
    p1, p2 = sum(r["p1"] for r in new), sum(r["p2"] for r in new)
    old = [r for r in res if r["model"] == "old"]
    summ = dict(P1_runs=p1, P2_runs=p2, P1_pass=p1 >= 27, P2_pass=p2 >= 27,
                P1_old=sum(r["p1"] for r in old), P2_old=sum(r["p2"] for r in old),
                P3={v: {mdl: float(np.median([r["picks_needed"] for r in res if r["version"] == v and r["model"] == mdl]))
                        for mdl in ("old", "new")} for v in VERSIONS},
                seconds=round(time.time() - t0))
    json.dump(dict(summary=summ, runs=res), open(os.path.join(OUT, "validate_gpc.json"), "w"), indent=1)
    for r in sorted(res, key=lambda r: (r["model"], r["version"], r["seed"])):
        print(f"{r['model']:3s} {r['version']:6s} seed {r['seed']}: picks needed {r['picks_needed']:3d}, final pick {r['final_pick']}, "
              f"wrong events at final {r['wrong_event_final']}, max 10-pick RMSE rise after 30: {r['max_rise']:+.3f} GPa, "
              f"P1 {'ok' if r['p1'] else 'FAIL'}, P2 {'ok' if r['p2'] else 'FAIL'}")
    print(json.dumps(summ, indent=1))
