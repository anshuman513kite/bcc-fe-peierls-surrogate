"""Third validation of EventGPD (V02_PLAN section 8.4, POST-HOC criterion change, 2 Nov 2026). SYNTHETIC, no LAMMPS.
FRESH seeds 120-129, surfaces weak / medium / strong, random selection on make_pool(2000), run protocol as 8.2 (>= 100 picks),
EventGPD only (30 runs). Reuses run() of v02_validate_gpd.py unchanged.
  P1': at the final pick, <= 1 T point with BOTH a wrong event AND |tau_P error| > 0.02 GPa, in >= 27 of 30 runs;
  P2 (unchanged): no RMSE rise > 0.05 GPa between consecutive 10-pick checks after pick 30, in >= 27 of 30 runs;
  P3: median picks needed per surface (descriptive); original P1 (<= 1 wrong-event point, any error): descriptive only.
Output: data/synthetic_v02/validate_gpd_p1prime.json. Usage (ml env): python v02_validate_gpd_p1prime.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v02_validate_gpd import run
from v02_pool_study import OUT, VERSIONS

SEEDS = list(range(120, 130))

if __name__ == "__main__":
    t0 = time.time()
    with Pool(7) as pool:
        res = pool.map(run, [(v, s, "EventGPD") for v in VERSIONS for s in SEEDS], chunksize=1)
    for r in res:
        r["p1prime"] = r["wrong_costly"] <= 1
    summ = dict(P1prime_runs=sum(r["p1prime"] for r in res), P2_runs=sum(r["p2"] for r in res),
                P1_original_runs_descriptive=sum(r["p1"] for r in res),
                median_picks={v: float(np.median([r["picks_needed"] for r in res if r["version"] == v])) for v in VERSIONS},
                seconds=round(time.time() - t0))
    summ["P1prime_pass"], summ["P2_pass"] = summ["P1prime_runs"] >= 27, summ["P2_runs"] >= 27
    json.dump(dict(summary=summ, runs=res), open(os.path.join(OUT, "validate_gpd_p1prime.json"), "w"), indent=1)
    for r in sorted(res, key=lambda r: (r["version"], r["seed"])):
        print(f"EventGPD {r['version']:6s} seed {r['seed']}: picks needed {r['picks_needed']:3d}, final pick {r['final_pick']}, "
              f"wrong events {r['wrong_event_final']} (costly, |err| > 0.02: {r['wrong_costly']}), max rise {r['max_rise']:+.3f}, "
              f"P1' {'ok' if r['p1prime'] else 'FAIL'}, P2 {'ok' if r['p2'] else 'FAIL'}, original P1 {'ok' if r['p1'] else 'fail'}")
    print("\ncostly wrong-event points (wrong event and |tau_P error| > 0.02 GPa):")
    for r in res:
        c = [w for w in r["wrong_points"] if abs(w["err"]) > 0.02]
        if c:
            print(f"  {r['version']} seed {r['seed']}: " + ", ".join(f"({w['chi']:+.0f},{w['sig']:+.2f}) true {w['true']} p={w['p']:.2f} gap {w['gap']:.3f} err {w['err']:+.3f}" for w in c))
    print(json.dumps(summ, indent=1))
