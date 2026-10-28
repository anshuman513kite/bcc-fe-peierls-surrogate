"""Session 12, Part B steps 1-3: build data/surrogate_train_v1.csv from the REAL chi sweep, then leave-one-out cross-validation
of the four model variants (scripts/surrogate_common.py). Settings are fixed in advance; nothing is tuned on a left-out point.
Per variant: error per left-out angle, RMSE, coverage of +-1 / +-2 spreads (spread includes the 0.0029 GPa noise).
Also checks that the ratio models' tau_P predictions do not depend on the scale tau0 (so leaving out chi = 0 leaks nothing).
Outputs (Session 14: renamed, so a re-run cannot overwrite the D6 training set): data/surrogate_train_v1_gateBp.csv, data/surrogate_loo.csv, data/surrogate_loo_summary.json.
Run (ml env): OMP_NUM_THREADS=1 nice -n 19 python scripts/surrogate_loo.py
"""
import csv, json, os, time
import numpy as np
import torch
from surrogate_common import ROOT, SIGMA, MODELS, make_model
TAU0 = 1.2875                                       # Session 12 baseline (Gate B'); this script reproduces Session 12
TRAIN_V1 = os.path.join(ROOT, "data", "surrogate_train_v1_gateBp.csv")   # D6 (Session 14): the current v1 has B2 at chi = 0
torch.set_num_threads(1)

# ---- step 1: dataset ----
sweep = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "peierls_chi_sweep.csv"))) if r["cell"] == "small"]
rows = [dict(chi_deg=float(r["chi_deg"]), side=r["side"], tau_last_unmoved=float(r["tau_last_unmoved"]), tau_first_moved=float(r["tau_first_moved"]),
             tau_mid=round(float(r["tau_mid"]), 6), noise_sigma=round(SIGMA, 6), glide_plane=r["glide_plane"], source=f"data/{r['name']}.csv")
        for r in sweep]
rows.append(dict(chi_deg=0.0, side="-", tau_last_unmoved=1.285, tau_first_moved=1.290, tau_mid=1.2875, noise_sigma=round(SIGMA, 6),
                 glide_plane="(-110)", source="data/gateBp_small_np7.csv (Gate B'; DD bracket re-checked Session 12)"))
rows.sort(key=lambda r: r["chi_deg"])
with open(TRAIN_V1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
chi = np.array([r["chi_deg"] for r in rows]); tau = np.array([r["tau_mid"] for r in rows])
print(f"dataset: {len(chi)} points -> {TRAIN_V1}")
for c, t in zip(chi, tau):
    print(f"  chi {c:+5.0f}  tau_mid {t:.4f}  ratio r = {t * np.cos(np.radians(c)) / TAU0:.4f}")

# ---- steps 2-3: leave-one-out ----
X = chi[:, None]
out, summ = [], {}
for name in MODELS:
    t0 = time.time(); m_, s_ = [], []
    for i in range(len(chi)):
        keep = np.arange(len(chi)) != i
        mdl = make_model(name).fit(X[keep], tau[keep])
        m, s = mdl.predict(X[i:i + 1])
        m_.append(float(m[0])); s_.append(float(s[0]))
        out.append(dict(model=name, chi_deg=chi[i], tau_true=tau[i], pred=m[0], spread=s[0], err=m[0] - tau[i], z=(m[0] - tau[i]) / s[0]))
    m_, s_ = np.array(m_), np.array(s_); err = m_ - tau; z = np.abs(err) / s_
    inner = np.abs(chi) < 29          # interpolation only (the two end points are extrapolation in LOO)
    summ[name] = dict(rmse=float(np.sqrt(np.mean(err ** 2))), rmse_interior=float(np.sqrt(np.mean(err[inner] ** 2))),
                      max_abs_err=float(np.max(np.abs(err))), within_1=float(np.mean(z <= 1)), within_2=float(np.mean(z <= 2)),
                      mean_spread=float(np.mean(s_)), seconds=round(time.time() - t0, 1))
    print(f"\n{name}: LOO RMSE {summ[name]['rmse']:.4f} GPa (interior 9 pts {summ[name]['rmse_interior']:.4f}), max |err| {summ[name]['max_abs_err']:.4f}, "
          f"inside +-1 / +-2 spread: {summ[name]['within_1']:.2f} / {summ[name]['within_2']:.2f} (ideal 0.68 / 0.95), mean spread {summ[name]['mean_spread']:.4f}, {summ[name]['seconds']} s")
    for c, t, m, s in zip(chi, tau, m_, s_):
        print(f"  chi {c:+5.0f}: true {t:.4f} pred {m:.4f} +- {s:.4f}  err {m - t:+.4f}  |err|/spread {abs(m - t) / s:.2f}")

# scale check: ratio models with tau0 = 1.0 must give the same tau_P (left-out chi = 0 and chi = +30)
for name in ("gp_ratio", "mlp_ratio"):
    for i in (int(np.where(chi == 0)[0][0]), len(chi) - 1):
        keep = np.arange(len(chi)) != i
        a = make_model(name).fit(X[keep], tau[keep]).predict(X[i:i + 1])
        b = make_model(name, tau0=1.0).fit(X[keep], tau[keep]).predict(X[i:i + 1])
        print(f"tau0 scale check {name} chi {chi[i]:+.0f}: tau0 = {TAU0}: {a[0][0]:.6f} +- {a[1][0]:.6f}; tau0 = 1.0: {b[0][0]:.6f} +- {b[1][0]:.6f}")
        summ[name].setdefault("tau0_check", []).append(dict(chi=chi[i], diff_mean=float(abs(a[0][0] - b[0][0])), diff_spread=float(abs(a[1][0] - b[1][0]))))

best = min(MODELS, key=lambda n: summ[n]["rmse"])
summ["best_by_rmse"] = best
print(f"\nbest by LOO RMSE: {best}")
with open(os.path.join(ROOT, "data", "surrogate_loo.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
json.dump(summ, open(os.path.join(ROOT, "data", "surrogate_loo_summary.json"), "w"), indent=1)
