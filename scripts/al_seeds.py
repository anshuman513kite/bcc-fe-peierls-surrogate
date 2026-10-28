"""SYNTHETIC test (Session 11): active learning, largest spread vs random choice, 20 seeds, compared at equal numbers of points.

Same setup as run_synthetic.py (pool of 61 points on the synthetic curve, noise 0.01, 3 starting points, MLP ensemble with
bootstrap + random priors, 800 epochs). The stopping rule is switched off (threshold 0) so every run adds 10 points (n = 13);
for each seed both strategies start from the same 3 points. Reports RMSE on the pool at n = 6, 8, 10, 13: mean, std, and the
fraction of seeds where 'spread' beats 'random' (paired by seed). Output: figures/synthetic/al_20seeds.json and .png.
Run (ml env): OMP_NUM_THREADS=1 nice -n 19 python scripts/al_seeds.py
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from surrogate import MLPEnsemble, active_learning, synthetic_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "synthetic")
pool = np.linspace(-30, 30, 61); truth = synthetic_curve(pool)
orc = lambda x: synthetic_curve(np.atleast_1d(x), noise=0.01, seed=int(abs(x[0]) * 1000) % 2**31)[0]
mk = lambda seed: MLPEnsemble(epochs=800, seed=seed)
NS = [6, 8, 10, 13]; SEEDS = range(20)
res = {"spread": {}, "random": {}}
for seed in SEEDS:
    for strat in res:
        h = active_learning(orc, pool, mk, n_init=3, strategy=strat, threshold=0.0, max_steps=10, seed=seed, truth=truth)["hist"]
        res[strat][seed] = {r["n"]: r["rmse"] for r in h}
    print(f"seed {seed}: " + " | ".join(f"{s} n=13 rmse {res[s][seed][13]:.4f}" for s in res), flush=True)
summ = {}
for n in NS:
    sp = np.array([res["spread"][s][n] for s in SEEDS]); ra = np.array([res["random"][s][n] for s in SEEDS])
    summ[n] = dict(spread_mean=sp.mean(), spread_std=sp.std(ddof=1), random_mean=ra.mean(), random_std=ra.std(ddof=1),
                   frac_spread_better=float(np.mean(sp < ra)), n_seeds=len(sp))
    print(f"n = {n:2d}: spread {sp.mean():.4f} +- {sp.std(ddof=1):.4f} | random {ra.mean():.4f} +- {ra.std(ddof=1):.4f} | "
          f"spread better in {np.mean(sp < ra) * 100:.0f} % of seeds")
json.dump(dict(per_n=summ, raw=res), open(f"{OUT}/al_20seeds.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
for k, (strat, c) in enumerate([("spread", "#2a78d6"), ("random", "#eb6834")]):
    m = [summ[n][f"{strat}_mean"] for n in NS]; s = [summ[n][f"{strat}_std"] for n in NS]
    ax.errorbar(np.array(NS) + (k - 0.5) * 0.15, m, yerr=s, color=c, marker="o", capsize=3,
                label=f"{'largest spread' if strat == 'spread' else 'random'} (mean ± std, 20 seeds)")
ax.set_xlabel("number of training points"); ax.set_ylabel("RMSE on the pool (synthetic)")
ax.set_title("SYNTHETIC test: active learning vs random choice, 20 seeds", fontsize=9)
ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/al_20seeds.png")
