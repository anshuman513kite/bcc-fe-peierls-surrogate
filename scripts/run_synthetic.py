"""Track 2 test of the surrogate code on a SYNTHETIC 1D curve (src/surrogate/synthetic.py). Not LAMMPS data.

Fits the MLP ensemble and the GP on 8 noisy points, checks calibration on a dense grid, and runs active learning
(spread-based vs random, 3 seeds). Plots go to figures/synthetic/ (every title says SYNTHETIC).
Run (ml env): OMP_NUM_THREADS=1 python scripts/run_synthetic.py
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from surrogate import MLPEnsemble, GPBaseline, active_learning, compare_strategies, coverage, synthetic_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "synthetic")
os.makedirs(OUT, exist_ok=True)
NOISE = 0.01
xg = np.linspace(-30, 30, 121); yg = synthetic_curve(xg)
rng = np.random.default_rng(0)
xt = np.sort(rng.uniform(-30, 30, 8)); yt = synthetic_curve(xt, noise=NOISE, seed=1)
res = {}
C = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150, sharey=True)
for ax, (name, model) in zip(axs, [("MLP ensemble (5 members)", MLPEnsemble(epochs=1500)), ("GP baseline", GPBaseline())]):
    model.fit(xt[:, None], yt)
    m, s = model.predict(xg[:, None])
    cov = coverage(yg, m, s)
    res[name] = dict(rmse=float(np.sqrt(np.mean((m - yg) ** 2))), **cov)
    ax.fill_between(xg, m - 2 * s, m + 2 * s, color=C[0], alpha=0.15, label="±2 spread")
    ax.fill_between(xg, m - s, m + s, color=C[0], alpha=0.3, label="±1 spread")
    ax.plot(xg, yg, color="#1a1a19", lw=1, ls="--", label="true SYNTHETIC curve")
    ax.plot(xg, m, color=C[0], lw=2, label="prediction")
    ax.scatter(xt, yt, color=C[1], zorder=3, label=f"training points (noise {NOISE})")
    ax.set_title(f"SYNTHETIC test: {name}\nRMSE {res[name]['rmse']:.4f}; inside ±1/±2: {cov['within_1sigma']:.2f}/{cov['within_2sigma']:.2f}", fontsize=9)
    ax.set_xlabel("x (synthetic; stands in for chi, deg)")
axs[0].set_ylabel("y (synthetic)"); axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fit_ensemble_vs_gp.png")

pool = np.linspace(-30, 30, 61)
truth = synthetic_curve(pool)
orc = lambda x: synthetic_curve(np.atleast_1d(x), noise=NOISE, seed=int(abs(x[0]) * 1000) % 2**31)[0]
mk = lambda seed: MLPEnsemble(epochs=800, seed=seed)
cmp_ = compare_strategies(orc, pool, mk, truth, seeds=(0, 1, 2), n_init=3, threshold=0.02, max_steps=10)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
for k, (strat, runs) in enumerate(cmp_.items()):
    for j, h in enumerate(runs):
        ax.plot([r["n"] for r in h], [r["rmse"] for r in h], color=C[k], alpha=0.8, marker="o", ms=3,
                label=f"{'largest spread' if strat == 'spread' else 'random'}" if j == 0 else None)
ax.set_yscale("log"); ax.set_xlabel("number of training points"); ax.set_ylabel("RMSE on the pool (synthetic)")
ax.set_title("SYNTHETIC test: active learning, largest-spread vs random choice (3 seeds each)", fontsize=9)
ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/active_learning.png")
res["active_learning"] = {s: [dict(final_n=h[-1]["n"], final_rmse=h[-1]["rmse"], final_max_spread=h[-1]["max_spread"]) for h in runs]
                          for s, runs in cmp_.items()}
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
