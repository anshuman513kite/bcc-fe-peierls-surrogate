"""Session 12, Part B step 4: retrospective active learning on the 11-point REAL pool (data/surrogate_train_v1.csv).

The pool is the 11 measured angles; the 'oracle' returns the measured tau_P (no new runs). Start from 3 points (random per
seed, the same 3 for both strategies), then add the pool point with the largest predicted spread, or a random one, until all
11 are in. RMSE on all 11 points after each refit (n = 3..11; at n = 11 both strategies are identical by construction).
20 seeds. Model given on the command line (default: the LOO winner).
Outputs: data/al_retro_<model>.json (per seed and summary). Run (ml env): OMP_NUM_THREADS=1 nice -n 19 python scripts/al_retro.py [model]
"""
import json, os, sys
import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon
from surrogate_common import ROOT, load_train, make_model
sys.path.insert(0, os.path.join(ROOT, "src"))
from surrogate import active_learning
torch.set_num_threads(1)
name = sys.argv[1] if len(sys.argv) > 1 else json.load(open(os.path.join(ROOT, "data", "surrogate_loo_summary.json")))["best_by_rmse"]
chi, tau = load_train(include_live=False)
lookup = dict(zip(chi.round(3), tau))
orc = lambda x: lookup[round(float(x[0]), 3)]
mk = lambda seed: make_model(name, seed=seed)
res = {"spread": {}, "random": {}}
for seed in range(20):
    for strat in res:
        h = active_learning(orc, chi, mk, n_init=3, strategy=strat, threshold=0.0, max_steps=8, seed=seed, truth=tau)
        res[strat][seed] = dict(rmse={r["n"]: r["rmse"] for r in h["hist"]}, order=[float(chi[i]) for i in h["idx"]])
    print(f"seed {seed:2d}: start {res['spread'][seed]['order'][:3]} | spread order {res['spread'][seed]['order'][3:]} | "
          + " ".join(f"{s} rmse n=5/7/9 {res[s][seed]['rmse'][5]:.4f}/{res[s][seed]['rmse'][7]:.4f}/{res[s][seed]['rmse'][9]:.4f}" for s in res), flush=True)
summ = {}
print(f"\nmodel {name}: RMSE on all 11 points (GPa), mean +- std over 20 seeds")
print(" n | spread            | random            | spread better | sign p | mean paired diff (spread-random) | Wilcoxon p")
for n in range(3, 12):
    a = np.array([res["spread"][s]["rmse"][n] for s in range(20)]); b = np.array([res["random"][s]["rmse"][n] for s in range(20)])
    d = a - b; nz = d[np.abs(d) > 1e-12]; win = int(np.sum(d < -1e-12))
    p_sign = float(binomtest(win, len(nz)).pvalue) if len(nz) else 1.0
    p_w = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    summ[n] = dict(spread_mean=float(a.mean()), spread_std=float(a.std(ddof=1)), random_mean=float(b.mean()), random_std=float(b.std(ddof=1)),
                   spread_better=win, n_nonzero=len(nz), sign_p=p_sign, diff_mean=float(d.mean()), diff_se=float(d.std(ddof=1) / np.sqrt(20)), wilcoxon_p=p_w)
    print(f"{n:2d} | {a.mean():.4f} +- {a.std(ddof=1):.4f} | {b.mean():.4f} +- {b.std(ddof=1):.4f} | {win:2d}/{len(nz):2d} | {p_sign:.3f} | "
          f"{d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(20):.4f} | {p_w:.3f}")
json.dump(dict(model=name, per_seed=res, summary=summ), open(os.path.join(ROOT, "data", f"al_retro_{name}.json"), "w"), indent=1)
