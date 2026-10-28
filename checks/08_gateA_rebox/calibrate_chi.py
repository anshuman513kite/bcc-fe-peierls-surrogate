"""Gate A calibration of ONE new angle chi for live active learning (Session 12), through the rebox path.

Same method and pass criteria as checks/07_chi_calibration/calibrate.py (option 2, approved): perfect slab, surface forces
for sigma_xy = tau cos(chi), box strains (eps_xx, eps_zz, gamma_xz) for sigma_xz = tau sin(chi), starting from the linear
slab strains, up to 4 Newton corrections; calibrate at tau = 1.0, 1.5, 2.0, 2.4; correction = q2 tau^2 + q3 tau^3 + q4 tau^4
(least squares); check the rule without iteration at tau = 0.5, 1.25, 1.75, 2.2. Pass test: tools/gate_a.py (Gate A criteria
with Ansh's D2 for sigma_xz, Session 13: within 1 % of target OR loading-angle error <= 0.1 deg). Only difference: the strains are set by
tools/rebox.py (in.measure), not by change_box remap (Session 11 bug; recheck.py showed <= 7e-5 GPa difference on a slab).
Adds the chi entry to strain_model_al.json (a copy of ../07_chi_calibration/strain_model.json plus the new angles) and the
rows to calibration_al.csv. Exit status 1 if any point fails Gate A (the live run must then not start).
Usage (lammps env, NP set by the launcher): python calibrate_chi.py <chi_deg_integer>
"""
import csv, json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, "../../tools")
from rebox import rebox
from gate_a import gate_a_check
NP = os.environ.get("NP", "7")
CV = np.load("../03_tilt_xz/Crot.npy")
fr, fx = [1, 3, 5], [0, 2, 4]
K = CV[np.ix_(fx, fx)] - CV[np.ix_(fx, fr)] @ np.linalg.solve(CV[np.ix_(fr, fr)], CV[np.ix_(fr, fx)])
LIN = np.array([0.0, 0.0027625, 0.0164349])
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
TAUS_CAL, TAUS_CHECK = (1.0, 1.5, 2.0, 2.4), (0.5, 1.25, 1.75, 2.2)
chi = int(sys.argv[1])


def lmp(inp, log, **v):
    args = ["mpirun", "-np", NP, "lmp", "-in", inp, "-log", log, "-screen", "none"]
    for k, x in v.items():
        args += ["-var", k, str(x)]
    subprocess.run(args, check=True)
    return open(log).read()


def measure(tag, e, sxy):
    t = open("slab0.data").read()
    Lx0 = float(re.search(r"^0 (\S+) xlo xhi", t, re.M).group(1)); Lz0 = float(re.search(r"^0 (\S+) zlo zhi", t, re.M).group(1))
    Lz = Lz0 * (1 + e[1])
    rebox("slab0.data", f"{tag}.data", Lx0 * (1 + e[0]), Lz, e[2] * Lz)
    out = lmp("in.measure", f"log.{tag}", tag=tag, datain=f"{tag}.data", exx=f"{e[0]:.10f}", ezz=f"{e[1]:.10f}", sxy=f"{sxy:.10f}")
    os.remove(f"{tag}.data")
    d = dict(kv.split("=") for kv in re.search(r"^INTERIOR_GPa \S+ (.*)$", out, re.M).group(1).split())
    return np.array([float(d["s" + n]) for n in names])


def check(s, tau):
    return gate_a_check(s, chi, tau)


if not os.path.exists("slab0.data"):
    lmp("in.build", "log.build")
ref = measure("al_ref", np.zeros(3), 0.0)
rows, cal = [], {}
fmt = lambda v: " ".join(f"{n}={x:+.5f}" for n, x in zip(names, v))
for tau in TAUS_CAL:
    e = LIN * tau * np.sin(np.radians(chi)); sxy = tau * np.cos(np.radians(chi))
    for it in range(5):
        s = measure(f"al_chi{chi:+d}_tau{tau}_it{it}", e, sxy) - ref
        err, rxy, rxz, ang, oth, ok = check(s, tau)
        print(f"chi {chi:+3d} tau {tau} it {it}: {fmt(s)} | xy {rxy*100:.3f} % xz {rxz*100:.3f} % ({np.degrees(ang):.3f} deg) other {oth*100:.3f} % of tau | {'PASS' if ok else 'fail'}", flush=True)
        if (rxy < 0.001 and rxz < 0.001 and oth < 0.001) or it == 4:
            break
        e = e - np.linalg.solve(K, err[[0, 2, 4]]); sxy = sxy - err[5]
    cal[tau] = (e.copy(), sxy)
    rows.append(dict(chi=chi, tau=tau, kind="calibrated", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, **{f"s{n}": v for n, v in zip(names, s)},
                     err_xy_pct=rxy * 100, err_xz_pct=rxz * 100, err_xz_angle_deg=np.degrees(ang), other_pct_tau=oth * 100, gate_A="PASS" if ok else "FAIL", iterations=it,
                     rule="D2"))
M = np.array([[t ** 2, t ** 3, t ** 4] for t in TAUS_CAL])
corr = lambda tau: np.append(cal[tau][0] - LIN * tau * np.sin(np.radians(chi)), cal[tau][1] - tau * np.cos(np.radians(chi)))
q = np.linalg.lstsq(M, np.array([corr(t) for t in TAUS_CAL]), rcond=None)[0]
for tau in TAUS_CHECK:
    c = q[0] * tau ** 2 + q[1] * tau ** 3 + q[2] * tau ** 4
    e = LIN * tau * np.sin(np.radians(chi)) + c[:3]; sxy = tau * np.cos(np.radians(chi)) + c[3]
    s = measure(f"al_chi{chi:+d}_tau{tau}_model", e, sxy) - ref
    err, rxy, rxz, ang, oth, ok = check(s, tau)
    print(f"chi {chi:+3d} tau {tau} MODEL: {fmt(s)} | xy {rxy*100:.3f} % xz {rxz*100:.3f} % ({np.degrees(ang):.3f} deg) other {oth*100:.3f} % of tau | {'PASS' if ok else 'fail'}", flush=True)
    rows.append(dict(chi=chi, tau=tau, kind="model_check", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, **{f"s{n}": v for n, v in zip(names, s)},
                     err_xy_pct=rxy * 100, err_xz_pct=rxz * 100, err_xz_angle_deg=np.degrees(ang), other_pct_tau=oth * 100, gate_A="PASS" if ok else "FAIL", iterations=0,
                     rule="D2"))
for r in rows:
    r["gate_A_D2"] = r["gate_A"]
new = not os.path.exists("calibration_al.csv")
keys = list(rows[0]) if new else next(csv.reader(open("calibration_al.csv")))   # keep the file's column order (recheck_d2.py)
with open("calibration_al.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval="")
    if new:
        w.writeheader()
    w.writerows(rows)
passed = all(r["gate_A"] == "PASS" for r in rows)
if passed:
    path = "strain_model_al.json" if os.path.exists("strain_model_al.json") else "../07_chi_calibration/strain_model.json"
    m = json.load(open(path))
    m["correction"][str(chi)] = dict(q2=q[0].tolist(), q3=q[1].tolist(), q4=q[2].tolist())
    m["note_al"] = "Session 12: angles added for live active learning by calibrate_chi.py (rebox path); 6-degree angles copied from ../07_chi_calibration"
    json.dump(m, open("strain_model_al.json", "w"), indent=1)
print(f"GATE_A_CHI {chi:+d}: {'PASS' if passed else 'FAIL'} ({sum(r['gate_A'] == 'PASS' for r in rows)}/{len(rows)})")
sys.exit(0 if passed else 1)
