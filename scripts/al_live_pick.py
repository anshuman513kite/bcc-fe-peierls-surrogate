"""Live active learning, pick step (Session 12, Part B step 5). Pre-registers the prediction BEFORE the LAMMPS run.

Fits the live-AL model on the current training set (data/surrogate_train_v1.csv + D5 rows of data/al_live_results.csv).
Model: rounds 1-2 used the Session 12 LOO winner (gp_ratio, fixed noise); from round 3 on, gp_tau_d4 (Ansh's D4/D7, Session
13-14: chosen in A5 before the round-3 data, not re-selected). The model evaluates the candidate pool chi = -30..+30 (1 deg) minus the angles
already run, and picks the largest predicted spread. Appends one row to data/al_live_predictions.csv (round, time, model,
n_train, pick, mean, spread, +-2 band, max pool spread, runner-up candidates) and writes the pool predictions to
data/al_live_pool_round<k>.csv. If the round is already in the predictions file (and not voided), the recorded pick is reused, never rewritten.
--evaluate: only write the pool file (no pick), e.g. after the last round (data/al_live_pool_after<k>.csv).
Prints: PICK <round> <chi>
Usage (ml env): python scripts/al_live_pick.py <round> [--evaluate]
"""
import csv, json, os, sys, time
import numpy as np
import torch
from surrogate_common import ROOT, load_train, make_model
torch.set_num_threads(1)
k = int(sys.argv[1]); evaluate = "--evaluate" in sys.argv
PRED = os.path.join(ROOT, "data", "al_live_predictions.csv")
if not evaluate and os.path.exists(PRED):
    for r in csv.DictReader(open(PRED)):
        if int(r["round"]) == k and not r.get("status", "").startswith("voided"):
            print(f"round {k} already pre-registered at {r['time']}: chi = {r['chi_pick']} (reused)")
            print(f"PICK {k} {int(float(r['chi_pick']))}"); sys.exit(0)
name = "gp_tau_d4" if k >= 3 else json.load(open(os.path.join(ROOT, "data", "surrogate_loo_summary.json")))["best_by_rmse"]
chi, tau = load_train()
model = make_model(name).fit(chi[:, None], tau)
pool = np.array([c for c in range(-30, 31) if c not in set(np.round(chi).astype(int))], float)
m, s = model.predict(pool[:, None])
tag = f"after{k}" if evaluate else f"round{k}"
with open(os.path.join(ROOT, "data", f"al_live_pool_{tag}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chi_deg", "pred_mean", "pred_spread"]); w.writerows([[c, round(a, 6), round(b, 6)] for c, a, b in zip(pool, m, s)])
print(f"{tag}: model {name}, {len(chi)} training points, pool {len(pool)} angles, max spread {s.max():.5f} GPa at chi {pool[np.argmax(s)]:+.0f}")
if evaluate:
    sys.exit(0)
i = int(np.argmax(s)); o = np.argsort(-s)[1:4]
row = dict(round=k, time=time.strftime("%Y-%m-%d %H:%M:%S"), model=name, n_train=len(chi), chi_pick=int(pool[i]),
           pred_mean=round(float(m[i]), 5), pred_spread=round(float(s[i]), 5), band2_lo=round(float(m[i] - 2 * s[i]), 5),
           band2_hi=round(float(m[i] + 2 * s[i]), 5), max_pool_spread=round(float(s.max()), 5),
           runner_up=";".join(f"{pool[j]:+.0f}:{s[j]:.5f}" for j in o), status="valid")
new = not os.path.exists(PRED)
keys = list(row) if new else next(csv.reader(open(PRED)))
with open(PRED, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=""); (w.writeheader() if new else None); w.writerow(row)
print(f"PRE-REGISTERED round {k}: chi = {row['chi_pick']:+d} deg, predicted tau_P = {row['pred_mean']:.4f} +- {row['pred_spread']:.4f} GPa, "
      f"+-2 band {row['band2_lo']:.4f} .. {row['band2_hi']:.4f} GPa (runner-up {row['runner_up']})")
print(f"PICK {k} {row['chi_pick']}")
