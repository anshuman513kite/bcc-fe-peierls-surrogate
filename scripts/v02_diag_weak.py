"""Diagnosis (2 Nov 2026, Ansh's request): why does random selection need ~4x more picks on the WEAK synthetic surface than
on the medium one (seed 0: 102 vs 23)? SYNTHETIC data only, no LAMMPS. Uses the study's own code path
(scripts/v02_pool_study.py setup / fit_eval, EventGP, random_order) with the settings committed in 1435017, unchanged.
For weak and medium, seeds 0, 1, 2, random order on make_pool(2000), until RMSE <= 0.02 or 400 picks:
  RMSE on T every 10 picks; at pick 25 and at the final pick the 5 T points with the largest |error| (with both GP means);
  how many picks were split events and from which chi region; split undercuts on T (split predicted, truth glide).
Usage (ml env): python v02_diag_weak.py
"""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v02_pool_study import setup, fit_eval, LARGE_POOL, N_CAP, TARGET, N_INIT, T, XT
from surrogate.designs import random_order, scaled

T_BUDGET = 25 * 60
t_start = time.time()


def region(chi):
    return "near -30" if chi < -15 else ("near +27" if chi > 15 else "other")


def table(m, yT, evT, label):
    mu, _, pev = m.predict(XT)
    (mg, _), (ms, _) = m.predict_both(XT)
    err = mu - yT
    print(f"  {label}: 5 largest |error| on T")
    print("     chi    sig   true  ev_true    pred  ev_pred  glide_mean  split_mean     err")
    for k in np.argsort(-np.abs(err))[:5]:
        print(f"   {T[k,0]:+5.0f} {T[k,1]:+6.3f}  {yT[k]:.3f}  {evT[k]:7s}  {mu[k]:.3f}  {pev[k]:7s}  {mg[k]:10.3f}  {ms[k]:10.3f}  {err[k]:+.3f}")
    under = np.where((pev == "split") & (evT == "glide"))[0]
    print(f"  {label}: split undercuts on T (split predicted, truth glide): {len(under)}"
          + ("" if not len(under) else ": " + ", ".join(f"({T[k,0]:+.0f},{T[k,1]:+.2f}) glide {mg[k]:.3f} split {ms[k]:.3f} true {yT[k]:.3f}" for k in under)))
    wrong_g = np.where((pev == "glide") & (evT == "split"))[0]
    print(f"  {label}: glide predicted where truth split: {len(wrong_g)}"
          + ("" if not len(wrong_g) else ": " + ", ".join(f"({T[k,0]:+.0f},{T[k,1]:+.2f})" for k in wrong_g)))
    print(f"  {label}: model n_glide {m.n_glide}, n_split {m.n_split}; RMSE over T split points "
          f"{np.sqrt(np.mean(err[evT == 'split'] ** 2)):.4f}, over T glide points {np.sqrt(np.mean(err[evT == 'glide'] ** 2)):.4f}")
    return len(under)


summary = []
for version in ("weak", "medium"):
    P, nc, ns, y, ev, yT, evT = setup(version, LARGE_POOL)
    print(f"\n===== {version}: pool {len(P)} points ({nc} x {ns} + edges); T split points {int((evT == 'split').sum())} of 49 "
          f"(at chi = {sorted(set(T[evT == 'split', 0]))}); pool split points {int((ev == 'split').sum())}")
    for seed in (0, 1, 2):
        if time.time() - t_start > T_BUDGET:
            print("TIME BOX reached (25 min): stopping"); break
        order = random_order(len(P), seed)
        print(f"\n--- {version} seed {seed}")
        rmse_line, reached, und25 = [], None, None
        for n in range(N_INIT, N_CAP + 1):
            m, r, pev = fit_eval(P, y, ev, order[:n], yT, evT, seed)
            if n % 10 == 0:
                rmse_line.append(f"{n}:{r:.4f}")
            if n == 25:
                und25 = table(m, yT, evT, "pick 25")
            if r <= TARGET:
                reached = n; break
        print("  RMSE every 10 picks: " + " ".join(rmse_line))
        print(f"  reached RMSE <= {TARGET} at pick: {reached if reached else f'not within {N_CAP}'} (RMSE {r:.4f})")
        picked = order[:n]
        sp = [i for i in picked if ev[i] == "split"]
        reg = {k: sum(region(P[i, 0]) == k for i in sp) for k in ("near -30", "near +27", "other")}
        print(f"  picks that were split events: {len(sp)} of {n}: {reg}")
        undf = table(m, yT, evT, f"final pick {n}")
        summary.append((version, seed, reached, len(sp), und25, undf))
print("\nSUMMARY version seed picks_needed split_picks undercuts@25 undercuts@final")
for s in summary:
    print("  ", *s)
print(f"compute time {time.time() - t_start:.0f} s")
