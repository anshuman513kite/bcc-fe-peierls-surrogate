"""Validation of fix (d) (V02_PLAN section 8.3, 2 Nov 2026): P1, P2 (EventGPD) and P3 (EventGPC vs EventGPD) on FRESH seeds
110-119, surfaces weak / medium / strong, random selection on make_pool(2000), same random orders for both models (60 runs).
Run protocol as 8.2 (>= 100 picks; picks needed = first pick with RMSE <= 0.02, 401 if never; final pick = last pick).
Descriptive only (NOT a pass condition): wrong-event T points at the final pick with |tau_P error| > 0.02 GPa, and the
largest |tau_P error| at a wrong-event point. For EventGPD the wrong-event points of failing runs are listed (as
v02_p1_where.py). SYNTHETIC, no LAMMPS. Output: data/synthetic_v02/validate_gpd.json. Usage (ml env): python v02_validate_gpd.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time, warnings
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v02_pool_study import setup, LARGE_POOL, N_CAP, TARGET, N_INIT, XT, T, OUT, VERSIONS
from surrogate.event_gp import EventGPC, EventGPD
from surrogate.designs import random_order, scaled
from surrogate.synthetic2d import surfaces

SEEDS, MIN_PICKS, MAX_RISE = list(range(110, 120)), 100, 0.05
MODELS = {"EventGPC": EventGPC, "EventGPD": EventGPD}


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
    w = np.where(pev != evT)[0]
    err = mu - yT
    g, s = surfaces(T[:, 0], T[:, 1], version)
    p = m.p_split(XT)
    wrong = [dict(chi=float(T[k, 0]), sig=float(T[k, 1]), true=str(evT[k]), p=float(p[k]), gap=float(abs(g[k] - s[k])), err=float(err[k])) for k in w]
    return dict(version=version, seed=seed, model=model, picks_needed=reached or N_CAP + 1, final_pick=n, wrong_event_final=len(w),
                wrong_costly=int(sum(abs(err[k]) > 0.02 for k in w)), wrong_max_err=float(max(abs(err[w]))) if len(w) else 0.0,
                wrong_points=wrong, max_rise=max(rises) if rises else 0.0, rmse_final=rmse[n],
                p1=len(w) <= 1, p2=(max(rises) if rises else 0.0) <= MAX_RISE, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    t0 = time.time()
    tasks = [(v, s, mdl) for mdl in MODELS for v in VERSIONS for s in SEEDS]
    with Pool(7) as pool:
        res = pool.map(run, tasks, chunksize=1)
    summ = {}
    for mdl in MODELS:
        rr = [r for r in res if r["model"] == mdl]
        summ[mdl] = dict(P1_runs=sum(r["p1"] for r in rr), P2_runs=sum(r["p2"] for r in rr),
                         median_picks={v: float(np.median([r["picks_needed"] for r in rr if r["version"] == v])) for v in VERSIONS},
                         wrong_points_total=sum(r["wrong_event_final"] for r in rr),
                         wrong_points_err_gt_0p02=sum(r["wrong_costly"] for r in rr),
                         max_err_at_wrong_point=max(r["wrong_max_err"] for r in rr))
    summ["P1_pass"] = summ["EventGPD"]["P1_runs"] >= 27; summ["P2_pass"] = summ["EventGPD"]["P2_runs"] >= 27
    summ["seconds"] = round(time.time() - t0)
    json.dump(dict(summary=summ, runs=res), open(os.path.join(OUT, "validate_gpd.json"), "w"), indent=1)
    for r in sorted(res, key=lambda r: (r["model"], r["version"], r["seed"])):
        print(f"{r['model']} {r['version']:6s} seed {r['seed']}: picks needed {r['picks_needed']:3d}, final pick {r['final_pick']}, "
              f"wrong events {r['wrong_event_final']} ({r['wrong_costly']} with |err| > 0.02), max rise {r['max_rise']:+.3f}, "
              f"P1 {'ok' if r['p1'] else 'FAIL'}, P2 {'ok' if r['p2'] else 'FAIL'}")
    print("\nEventGPD wrong-event points in runs failing P1:")
    for r in res:
        if r["model"] == "EventGPD" and not r["p1"]:
            print(f"  {r['version']} seed {r['seed']}: RMSE {r['rmse_final']:.4f}; " + ", ".join(
                f"({w['chi']:+.0f},{w['sig']:+.2f}) true {w['true']} p={w['p']:.2f} gap {w['gap']:.3f} err {w['err']:+.3f}" for w in r["wrong_points"]))
    print(json.dumps(summ, indent=1))
