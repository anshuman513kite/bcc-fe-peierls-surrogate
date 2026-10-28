"""Session 13, A4: Vitek-type benchmark of the chi curve against Chaussidon, Fivel & Rodney, Acta Mater. 54, 3407 (2006)
(same potential: Mendelev 2003 #2, free boundaries; their fit s0 = 1604 MPa, a = 0.61, and 1210 MPa measured at chi = 0,
as read by the planning chat; the paper itself was not read here).

Model (Vitek-type, glide on the (-110) plane): tau(chi) = s0 / (cos chi + a cos(60 deg + chi)).
  a = 0 is Schmid's law; a > 0 makes chi > 0 harder. Assumption, not checked against the paper: their chi has the same
  sign convention as ours (chi > 0 = anti-twinning side), since a > 0 then means anti-twinning harder.
Fitted by least squares (scipy curve_fit, equal weights) to OUR points whose glide plane is (-110) (printed); points that
moved on another {110} plane, or whose value is pending (core-change events), are not used. Standard errors from the fit
covariance scaled by the residual variance.
Outputs: data/vitek_fit.json, data/vitek_compare.csv. Run (ml env): python scripts/fit_vitek.py
"""
import csv, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CHAUSSIDON = dict(s0=1.604, a=0.61)          # GPa, Chaussidon et al. 2006 fit (planning chat)
CHAUSSIDON_CHI0 = 1.210                       # GPa, measured at chi = 0 (text, Section 4; planning chat)


def vitek_tau(chi_deg, s0, a):
    c = np.radians(np.asarray(chi_deg, float))
    return s0 / (np.cos(c) + a * np.cos(np.radians(60.0) + c))


def our_points():
    """(chi, tau_mid, glide plane, source) of every measured small-cell point, chi = 0 and the D1 live rounds."""
    pts = [(float(r["chi_deg"]), float(r["tau_mid"]), r["glide_plane"], r["source"]) for r in csv.DictReader(open(f"{ROOT}/data/surrogate_train_v1.csv"))]
    live = f"{ROOT}/data/al_live_results.csv"
    if os.path.exists(live):
        pts += [(float(r["chi_deg"]), float(r["tau_mid"]), r["glide_plane"], f"live round {r['round']}") for r in csv.DictReader(open(live)) if r.get("rule") == "D5"]
    return sorted(pts)


if __name__ == "__main__":
    from scipy.optimize import curve_fit
    pts = our_points()
    use = [p for p in pts if p[2] == "(-110)"]
    print("points used (glide plane (-110)):", ", ".join(f"{p[0]:+.0f}" for p in use))
    print("not used:", ", ".join(f"{p[0]:+.0f} ({p[2]})" for p in pts if p not in use))
    x = np.array([p[0] for p in use]); y = np.array([p[1] for p in use])
    (s0, a), cov = curve_fit(vitek_tau, x, y, p0=(1.6, 0.6))
    se = np.sqrt(np.diag(cov)); res = y - vitek_tau(x, s0, a)
    print(f"fit: s0 = {s0 * 1000:.1f} +- {se[0] * 1000:.1f} MPa, a = {a:.4f} +- {se[1]:.4f}; RMS residual {np.sqrt(np.mean(res ** 2)) * 1000:.1f} MPa; "
          f"tau(0) from fit {vitek_tau(0, s0, a) * 1000:.1f} MPa")
    print(f"Chaussidon fit: s0 = {CHAUSSIDON['s0'] * 1000:.0f} MPa, a = {CHAUSSIDON['a']}; tau(0) from their fit {vitek_tau(0, **CHAUSSIDON) * 1000:.1f} MPa; "
          f"their measured chi = 0: {CHAUSSIDON_CHI0 * 1000:.0f} MPa")
    rows = []
    print(f"\n{'chi':>5s} {'plane':10s} {'ours':>7s} {'our fit':>8s} {'resid':>7s} {'Chaussidon fit':>15s} {'ours/theirs':>11s}")
    for c, t, pl, src in pts:
        ours_fit = vitek_tau(c, s0, a); th = vitek_tau(c, **CHAUSSIDON)
        r = t - ours_fit
        rows.append(dict(chi_deg=c, glide_plane=pl, used_in_fit="yes" if pl == "(-110)" else "no", tau_ours=t, tau_our_fit=round(float(ours_fit), 5),
                         residual=round(float(r), 5), tau_chaussidon_fit=round(float(th), 5), ratio_ours_theirs=round(t / float(th), 4), source=src))
        print(f"{c:+5.0f} {pl:10s} {t:7.4f} {ours_fit:8.4f} {r:+7.4f} {th:15.4f} {t / th:11.3f}")
    json.dump(dict(points_used=[p[0] for p in use], s0_GPa=s0, s0_se_GPa=se[0], a=a, a_se=se[1], rms_residual_GPa=float(np.sqrt(np.mean(res ** 2))),
                   tau0_from_fit_GPa=float(vitek_tau(0, s0, a)), chaussidon=CHAUSSIDON, chaussidon_tau0_fit=float(vitek_tau(0, **CHAUSSIDON)),
                   chaussidon_chi0_measured=CHAUSSIDON_CHI0), open(f"{ROOT}/data/vitek_fit.json", "w"), indent=1)
    with open(f"{ROOT}/data/vitek_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
