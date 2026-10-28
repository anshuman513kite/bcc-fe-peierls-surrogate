"""Session 13, A5: leave-one-out with Ansh's D4 GP (noise learned within [0.0029, 0.1] GPa): gp_tau_d4, gp_ratio_d4, and the
Session 12 fixed-noise gp_ratio for comparison. Point sets: 'all' = v1 (11) + live rounds recorded under D1 (rule = D1 in
data/al_live_results.csv); '(-110) only' = the subset whose glide plane is (-110) (diagnostic).
Reported: RMSE, coverage of +-1 / +-2 spreads, learned noise (GPa; median over the LOO folds, and on the full set).
Model for the rest of live AL (fixed here, before the runs): the D4 model with the lower LOO RMSE on 'all'.
Outputs: data/surrogate_loo_d4.csv, data/surrogate_loo_d4_summary.json; with a tag argument (Session 14, e.g. on the corrected
data after D5/D6): data/surrogate_loo_d4_<tag>.* (the A5 record and its model choice stay untouched).
Run (ml env): python scripts/surrogate_loo_d4.py [tag]
"""
import csv, json, os, sys
import numpy as np
from surrogate_common import ROOT, make_model, learned_noise_gpa
from fit_vitek import our_points
MODELS = ("gp_tau_d4", "gp_ratio_d4", "gp_ratio")
pts = our_points()
sets = {"all": pts, "(-110) only": [p for p in pts if p[2] == "(-110)"]}
out, summ = [], {}
for sname, P in sets.items():
    chi = np.array([p[0] for p in P]); tau = np.array([p[1] for p in P]); X = chi[:, None]
    print(f"\n=== {sname}: {len(P)} points: {', '.join(f'{c:+.0f}' for c in chi)}")
    for name in MODELS:
        m_, s_, nz = [], [], []
        for i in range(len(chi)):
            keep = np.arange(len(chi)) != i
            mdl = make_model(name).fit(X[keep], tau[keep]); m, s = mdl.predict(X[i:i + 1])
            m_.append(m[0]); s_.append(s[0]); nz.append(learned_noise_gpa(mdl))
            out.append(dict(set=sname, model=name, chi_deg=chi[i], tau_true=tau[i], pred=m[0], spread=s[0], err=m[0] - tau[i], noise_GPa=nz[-1]))
        m_, s_ = np.array(m_), np.array(s_); err = m_ - tau; z = np.abs(err) / s_
        full = learned_noise_gpa(make_model(name).fit(X, tau))
        summ[f"{sname}|{name}"] = r = dict(n=len(chi), rmse=float(np.sqrt(np.mean(err ** 2))), within_1=float(np.mean(z <= 1)), within_2=float(np.mean(z <= 2)),
                                          noise_median_GPa=float(np.median(nz)), noise_full_GPa=full, mean_spread=float(np.mean(s_)))
        print(f"{name:12s}: RMSE {r['rmse']:.4f} GPa | +-1/+-2 {r['within_1']:.2f}/{r['within_2']:.2f} | learned noise {r['noise_full_GPa']:.4f} GPa "
              f"(full set; LOO median {r['noise_median_GPa']:.4f}) | mean spread {r['mean_spread']:.4f}")
        print("   " + " ".join(f"{c:+.0f}:{e:+.3f}({zz:.1f})" for c, e, zz in zip(chi, err, z)))
TAG = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
best = min(("gp_tau_d4", "gp_ratio_d4"), key=lambda n: summ[f"all|{n}"]["rmse"])
summ["best_d4_by_rmse_all"] = best
print(f"\nlower LOO RMSE of the two D4 models: {best}" + (" (information only; the live-AL model is fixed by D7)" if TAG else " -> model for the rest of live AL"))
with open(f"{ROOT}/data/surrogate_loo_d4{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
json.dump(summ, open(f"{ROOT}/data/surrogate_loo_d4{TAG}_summary.json", "w"), indent=1)
