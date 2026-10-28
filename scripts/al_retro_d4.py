"""Session 13, A6: retrospective active learning with the D4 model (data/surrogate_loo_d4_summary.json: best_d4_by_rmse_all),
on the measured pool of A5 ('all': v1 + D1 live rounds). Oracle = the measured value. RMSE on the whole pool after each refit.
Strategies:
  spread  - 3 random starting points per seed, then the largest predicted spread;
  random  - the same 3 starting points, then random choice (20 seeds);
  bisect  - fixed rule, no randomness: start -30, 0, +30; then the pool point nearest the midpoint of the largest gap between
            chosen angles (ties: leftmost gap, lower chi);
  spread_fixedstart - the spread rule started from -30, 0, +30 (deterministic), to separate the rule from the starting points.
Reported: RMSE at n = 5..9 (mean, std over seeds), wins per seed of spread vs random and vs bisect.
Output: data/al_retro_d4.json. Run (ml env): OMP_NUM_THREADS=1 nice -n 19 python scripts/al_retro_d4.py
"""
import json, os
import numpy as np
from surrogate_common import ROOT, make_model
from fit_vitek import our_points
name = json.load(open(f"{ROOT}/data/surrogate_loo_d4_summary.json"))["best_d4_by_rmse_all"]
P = our_points(); chi = np.array([p[0] for p in P]); tau = np.array([p[1] for p in P]); N = len(chi)
NS = range(5, 10)


def rmse_after(idx):
    m, _ = make_model(name).fit(chi[idx, None], tau[idx]).predict(chi[:, None])
    return float(np.sqrt(np.mean((m - tau) ** 2)))


def run(start, pick, nmax=9):
    idx, res = list(start), {}
    while True:
        res[len(idx)] = rmse_after(idx)
        if len(idx) >= nmax:
            return res, [float(chi[i]) for i in idx]
        idx.append(pick(idx))


def spread_pick(idx):
    _, s = make_model(name).fit(chi[idx, None], tau[idx]).predict(chi[:, None])
    free = [i for i in range(N) if i not in idx]
    return free[int(np.argmax(s[free]))]


def bisect_pick(idx):
    sel = sorted(chi[idx])
    gaps = sorted(((sel[j + 1] - sel[j], -sel[j], sel[j], sel[j + 1]) for j in range(len(sel) - 1)), reverse=True)
    for _, _, lo, hi in gaps:
        inside = [i for i in range(N) if i not in idx and lo < chi[i] < hi]
        if inside:
            mid = 0.5 * (lo + hi)
            return min(inside, key=lambda i: (abs(chi[i] - mid), chi[i]))


FIXED = [int(np.where(chi == c)[0][0]) for c in (-30.0, 0.0, 30.0)]
bis, bis_order = run(FIXED, bisect_pick)
sfs, sfs_order = run(FIXED, spread_pick)
print(f"model {name}, pool {N} points")
print(f"bisect order: {bis_order}\nspread from -30/0/+30 order: {sfs_order}")
res = {"spread": {}, "random": {}}
for seed in range(20):
    rng = np.random.default_rng(seed)
    start = list(rng.choice(N, 3, replace=False))
    res["spread"][seed] = run(start, spread_pick)[0]
    rr = np.random.default_rng(1000 + seed)
    res["random"][seed] = run(start, lambda idx: int(rr.choice([i for i in range(N) if i not in idx])))[0]
summ = {}
print(f"\n n | spread (20 seeds)  | random (20 seeds)  | bisect | spread from fixed start | spread<random | spread<bisect")
for n in NS:
    a = np.array([res["spread"][s][n] for s in range(20)]); b = np.array([res["random"][s][n] for s in range(20)])
    summ[n] = dict(spread_mean=float(a.mean()), spread_std=float(a.std(ddof=1)), random_mean=float(b.mean()), random_std=float(b.std(ddof=1)),
                   bisect=bis[n], spread_fixedstart=sfs[n], spread_beats_random=int(np.sum(a < b - 1e-12)), spread_beats_bisect=int(np.sum(a < bis[n] - 1e-12)))
    print(f"{n:2d} | {a.mean():.4f} +- {a.std(ddof=1):.4f} | {b.mean():.4f} +- {b.std(ddof=1):.4f} | {bis[n]:.4f} | {sfs[n]:.4f}                  | "
          f"{summ[n]['spread_beats_random']:2d}/20         | {summ[n]['spread_beats_bisect']:2d}/20")
json.dump(dict(model=name, pool=chi.tolist(), bisect=bis, bisect_order=bis_order, spread_fixedstart=sfs, spread_fixedstart_order=sfs_order,
               per_seed=res, summary=summ), open(f"{ROOT}/data/al_retro_d4.json", "w"), indent=1)
