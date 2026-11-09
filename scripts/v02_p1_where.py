"""Where are the wrong-event T points of the runs that failed P1 (V02_PLAN 8.2) with the new model? Refits only the final
pick of each failing run (data/synthetic_v02/validate_gpc.json); prints, per wrong point, the true glide-split gap
|tau_g - tau_s| (unquantised) and the tau_P error there. SYNTHETIC; settings unchanged. Usage (ml env): python v02_p1_where.py"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v02_pool_study import setup, LARGE_POOL, XT, T, OUT
from surrogate.event_gp import EventGPC
from surrogate.designs import random_order, scaled
from surrogate.synthetic2d import surfaces
res = json.load(open(os.path.join(OUT, "validate_gpc.json")))["runs"]
gaps, errs, allpts = [], [], []
for r in res:
    if r["model"] != "new" or r["p1"]:
        continue
    P, nc, ns, y, ev, yT, evT = setup(r["version"], LARGE_POOL)
    o = random_order(len(P), r["seed"])[:r["final_pick"]]
    m = EventGPC(r["seed"]).fit(scaled(P[o]), y[o], ev[o])
    mu, _, pev = m.predict(XT); p = m.p_split(XT)
    g, s = surfaces(T[:, 0], T[:, 1], r["version"])
    w = np.where(pev != evT)[0]
    print(f"{r['version']:6s} seed {r['seed']}: RMSE {np.sqrt(np.mean((mu - yT) ** 2)):.4f}; wrong at " + ", ".join(
        f"({T[k,0]:+.0f},{T[k,1]:+.2f}) true {evT[k]} p={p[k]:.2f} gap {abs(g[k]-s[k]):.3f} err {mu[k]-yT[k]:+.3f}" for k in w))
    gaps += [abs(g[k] - s[k]) for k in w]; errs += [abs(mu[k] - yT[k]) for k in w]
gaps, errs = np.array(gaps), np.array(errs)
print(f"wrong-event points: {len(gaps)}; true glide-split gap: median {np.median(gaps):.3f}, max {gaps.max():.3f} GPa, "
      f"{(gaps < 0.02).sum()} below 0.02 GPa; |tau_P error| there: median {np.median(errs):.3f}, max {errs.max():.3f} GPa, "
      f"{(errs < 0.02).sum()} below 0.02 GPa")
