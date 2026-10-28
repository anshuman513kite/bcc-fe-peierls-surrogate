"""Gate A: calibrate the chi != 0 loading (option 2, provisional) on the perfect slab (checks/04_pure_shear/in.block, geom = slab).

Loading for (chi, tau): surface forces for sigma_xy = tau*cos(chi); box strains eps_xx, eps_zz, gamma_xz for sigma_xz = tau*sin(chi)
(y faces free). Start from the linear slab strains (per GPa of sigma_xz: eps_zz = +0.0027625, gamma_xz = +0.0164349); then up to
4 Newton corrections from the measured interior stress (minus the unloaded slab): box strains by the slab box-strain stiffness K
(components xx, zz, xz), surface-force stress by its own residual.
Gate A pass per (chi, tau): sigma_xy and sigma_xz within 1 % of their targets, every other component below 2 % of tau.
Rule for other tau (v2): calibrate at tau = 1.0, 1.5, 2.0, 2.4; correction (calibrated - linear) = q2*tau^2 + q3*tau^3 + q4*tau^4,
least squares per chi and quantity; validated without iteration at tau = 0.5, 1.25, 1.75, 2.2.
(v1 used tau = 1.0, 1.5 and q2, q3 only; its tau = 2.4 check failed at chi = +-6: calibration_v1.csv.)
Writes calibration.csv (all (chi, tau) incl. validation) and strain_model.json (linear coefficients + q2, q3 per chi).
Run (crash-safe): ../../tools/launch_safe.sh chi_calibration python calibrate.py
"""
import json, os, re, subprocess
import numpy as np
NP = os.environ.get("NP", "7")
CV = np.load("../03_tilt_xz/Crot.npy")
fr, fx = [1, 3, 5], [0, 2, 4]
K = CV[np.ix_(fx, fx)] - CV[np.ix_(fx, fr)] @ np.linalg.solve(CV[np.ix_(fr, fr)], CV[np.ix_(fr, fx)])   # (xx, zz, xz) box -> stress
LIN = np.array([0.0, 0.0027625, 0.0164349])           # (eps_xx, eps_zz, gamma_xz) per GPa of pure sigma_xz, slab
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
CHIS = [-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30]
TAUS_CAL, TAUS_CHECK = (1.0, 1.5, 2.0, 2.4), (0.5, 1.25, 1.75, 2.2)


def run(tag, e, sxy):
    log = f"log.{tag}"
    subprocess.run(["mpirun", "-np", NP, "lmp", "-in", "../04_pure_shear/in.block", "-log", log, "-screen", "none",
                    "-var", "tag", tag, "-var", "geom", "slab", "-var", "exx", f"{e[0]:.10f}", "-var", "eyy", "0",
                    "-var", "ezz", f"{e[1]:.10f}", "-var", "gyz", "0", "-var", "gxz", f"{e[2]:.10f}", "-var", "gxy", "0",
                    "-var", "sxy", f"{sxy:.10f}"], check=True)
    d = dict(kv.split("=") for kv in re.search(r"^INTERIOR_GPa \S+ (.*)$", open(log).read(), re.M).group(1).split())
    return np.array([float(d["s" + n]) for n in names])


def check(s, chi, tau):
    t = np.zeros(6); t[5] = tau * np.cos(np.radians(chi)); t[4] = tau * np.sin(np.radians(chi))
    err = s - t
    rel_xy = abs(err[5]) / abs(t[5])
    rel_xz = abs(err[4]) / abs(t[4]) if abs(t[4]) > 1e-12 else abs(err[4]) / tau
    other = max(abs(err[[0, 1, 2, 3]])) / tau
    ok = rel_xy < 0.01 and rel_xz < 0.01 and other < 0.02
    return t, err, rel_xy, rel_xz, other, ok


ref = run("slab_ref", np.zeros(3), 0.0)
rows, model = [], {}
fmt = lambda v: " ".join(f"{n}={x:+.5f}" for n, x in zip(names, v))
for chi in CHIS:
    cal = {}
    for tau in TAUS_CAL:
        sxz = tau * np.sin(np.radians(chi)); e = LIN * sxz; sxy = tau * np.cos(np.radians(chi))
        for it in range(5):
            s = run(f"chi{chi:+d}_tau{tau}_it{it}", e, sxy) - ref
            t, err, rxy, rxz, oth, ok = check(s, chi, tau)
            print(f"chi {chi:+3d} tau {tau} it {it}: {fmt(s)} | xy {rxy*100:.3f} % xz {rxz*100:.3f} % other {oth*100:.3f} % of tau | {'PASS' if ok else 'fail'}", flush=True)
            if (rxy < 0.001 and rxz < 0.001 and oth < 0.001) or it == 4:
                break
            e = e - np.linalg.solve(K, err[[0, 2, 4]]); sxy = sxy - err[5]
        cal[tau] = (e.copy(), sxy)
        rows.append(dict(chi=chi, tau=tau, kind="calibrated", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, **{f"s{n}": v for n, v in zip(names, s)},
                         err_xy_pct=rxy * 100, err_xz_pct=rxz * 100, other_pct_tau=oth * 100, gate_A="PASS" if ok else "FAIL", iterations=it))
    # correction model q2*tau^2 + q3*tau^3 through tau = 1.0, 1.5 (for e and for the surface-force stress)
    M = np.array([[t ** 2, t ** 3, t ** 4] for t in TAUS_CAL])
    corr = lambda tau: np.append(cal[tau][0] - LIN * tau * np.sin(np.radians(chi)), cal[tau][1] - tau * np.cos(np.radians(chi)))
    q = np.linalg.lstsq(M, np.array([corr(t) for t in TAUS_CAL]), rcond=None)[0]   # rows q2, q3, q4; cols exx, ezz, gxz, sxy
    model[chi] = dict(q2=q[0].tolist(), q3=q[1].tolist(), q4=q[2].tolist())
    for tau in TAUS_CHECK:
        c = q[0] * tau ** 2 + q[1] * tau ** 3 + q[2] * tau ** 4
        e = LIN * tau * np.sin(np.radians(chi)) + c[:3]; sxy = tau * np.cos(np.radians(chi)) + c[3]
        s = run(f"chi{chi:+d}_tau{tau}_model", e, sxy) - ref
        t, err, rxy, rxz, oth, ok = check(s, chi, tau)
        print(f"chi {chi:+3d} tau {tau} MODEL: {fmt(s)} | xy {rxy*100:.3f} % xz {rxz*100:.3f} % other {oth*100:.3f} % of tau | {'PASS' if ok else 'fail'}", flush=True)
        rows.append(dict(chi=chi, tau=tau, kind="model_check", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, **{f"s{n}": v for n, v in zip(names, s)},
                         err_xy_pct=rxy * 100, err_xz_pct=rxz * 100, other_pct_tau=oth * 100, gate_A="PASS" if ok else "FAIL", iterations=0))
import csv
with open("calibration.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(dict(lin_per_GPa_sxz=dict(exx=LIN[0], ezz=LIN[1], gxz=LIN[2]),
               note="strains(chi,tau) = LIN*tau*sin(chi) + q2*tau^2 + q3*tau^3 + q4*tau^4 (cols exx, ezz, gxz); sxy_applied = tau*cos(chi) + same polynomial with col 3",
               correction={str(k): v for k, v in model.items()}), open("strain_model.json", "w"), indent=1)
cal_ok = all(r["gate_A"] == "PASS" for r in rows if r["kind"] == "calibrated")
mod_ok = all(r["gate_A"] == "PASS" for r in rows if r["kind"] == "model_check")
print(f"GATE_A calibrated points: {'PASS' if cal_ok else 'FAIL'} ; model check at tau 0.5 / 1.25 / 1.75 / 2.2: {'PASS' if mod_ok else 'FAIL'}")
