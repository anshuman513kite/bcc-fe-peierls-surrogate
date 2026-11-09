"""v0.2 Gate A-yz calibration of ONE (chi, sigma_yz) pair on the perfect slab (V02_PLAN section 5.3 part 1, section 8 Q3/Q5).

Method as checks/08_gateA_rebox/calibrate_chi.py (option 2, rebox path), extended:
  - sigma_yz by surface forces along z on the same slabs (in.measure_yz); linear start: surface stress = sigma_yz, no box
    strain (linear-elastic calculation, checks/09_yz_coupling: sigma_yz adds no other stress at fixed box strains);
  - Newton corrections of the box strains (eps_xx, eps_zz, gamma_xz; slab box-strain stiffness K) and of both surface
    stresses, up to 4, from the measured interior stress;
  - calibrated tau at sigma_yz = 0: 1.0, 1.5, 2.0, 2.4, 2.7, 3.0 GPa (ramp top 3.0, Ansh's Q5); model checks without iteration
    at 0.5, 1.25, 1.75, 2.2, 2.55, 2.85 GPa;
  - at sigma_yz != 0 (V02_PLAN 8.7, 3 Nov 2026): calibrated tau 0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0 GPa; model checks 0.25, 0.75,
    1.25, 1.75, 2.2, 2.55, 2.85 GPa. The tau = 0.5 point was added after the 2 Nov smoke test: with only tau = 0 below 1.0 GPa the
    quartic bent in that gap (loading-angle error 0.377 deg at tau = 0.5);
  - correction model per component: q2 tau^2 + q3 tau^3 + q4 tau^4 at sigma_yz = 0 (v0.1 form); q0 + q1 tau + ... + q4 tau^4
    at sigma_yz != 0 (tau * sigma_yz cross terms are linear in tau, sigma_yz^2 terms constant). 5 components:
    (d eps_xx, d eps_zz, d gamma_xz, d sigma_xy surface stress, d sigma_yz surface stress);
  - pass test: tools/gate_a.py gate_a3_check (three loaded components, Q3 option a).
Writes calibration_yz.csv (appends) and, if every point passes, the entry "<chi>,<syz>" in strain_model_yz.json
(use with tools/peierls_chi.py --model ../../checks/10_gateA_yz/strain_model_yz.json --model_key "<chi>,<syz>").
Exit status 1 if any point fails. Usage (lammps env, NP from the launcher): python calibrate_yz.py <chi_deg> <sigma_yz_GPa>
The logic (calibrate()) takes the measurement as a function, so tests run it on a fake slab without LAMMPS.
"""
import csv, json, os, re, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../tools"))
from gate_a import gate_a3_check

LIN = np.array([0.0, 0.0027625, 0.0164349])          # (eps_xx, eps_zz, gamma_xz) per GPa of pure sigma_xz, slab (Session 9)
NAMES = ["xx", "yy", "zz", "yz", "xz", "xy"]
TAUS_CAL = (1.0, 1.5, 2.0, 2.4, 2.7, 3.0)                    # sigma_yz = 0
TAUS_CHECK = (0.5, 1.25, 1.75, 2.2, 2.55, 2.85)
TAUS_CAL_YZ = (0.0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0)          # sigma_yz != 0 (V02_PLAN 8.7)
TAUS_CHECK_YZ = (0.25, 0.75, 1.25, 1.75, 2.2, 2.55, 2.85)


def ladders(syz):
    """(calibrated tau, model-check tau) for this sigma_yz: section 8.1 ladder at sigma_yz = 0, section 8.7 ladder otherwise."""
    return (TAUS_CAL, TAUS_CHECK) if abs(syz) < 1e-12 else (TAUS_CAL_YZ, TAUS_CHECK_YZ)


def slab_stiffness():
    """3x3 box-strain stiffness of the slab (free y faces), from the rotated C' (checks/03_tilt_xz/Crot.npy)."""
    CV = np.load(os.path.join(HERE, "../03_tilt_xz/Crot.npy"))
    fr, fx = [1, 3, 5], [0, 2, 4]
    return CV[np.ix_(fx, fx)] - CV[np.ix_(fx, fr)] @ np.linalg.solve(CV[np.ix_(fr, fr)], CV[np.ix_(fr, fx)])


def powers(syz):
    return (2, 3, 4) if abs(syz) < 1e-12 else (0, 1, 2, 3, 4)


def calibrate(chi, syz, measure, K, taus_cal=None, taus_check=None, max_it=4, log=print):
    """measure(tag, e(3), sxy_applied, syz_applied) -> interior stress (xx, yy, zz, yz, xz, xy) minus the unloaded slab, GPa.
    Returns rows, entry (strain-model dict) and passed."""
    c, s_ = np.cos(np.radians(chi)), np.sin(np.radians(chi))
    lc, lk = ladders(syz)
    taus = tuple(lc if taus_cal is None else taus_cal)
    taus_check = tuple(lk if taus_check is None else taus_check)
    rows, cal = [], {}
    fmt = lambda v: " ".join(f"{n}={x:+.5f}" for n, x in zip(NAMES, v))
    for tau in taus:
        e, sxy, sy = LIN * tau * s_, tau * c, syz
        for it in range(max_it + 1):
            s = measure(f"yz_chi{chi:+g}_syz{syz:+g}_tau{tau}_it{it}", e, sxy, sy)
            g = gate_a3_check(s, chi, tau, syz)
            log(f"chi {chi:+g} syz {syz:+g} tau {tau} it {it}: {fmt(s)} | ok {g['ok']} (xy {g['ok_xy']}, xz {g['ok_xz']}, yz {g['ok_yz']}, other {g['oth']*100:.3f} %)")
            ref = max(tau, abs(syz))
            if np.max(np.abs(g["err"])) / ref < 0.001 or it == max_it:
                break
            e = e - np.linalg.solve(K, g["err"][[0, 2, 4]]); sxy = sxy - g["err"][5]; sy = sy - g["err"][3]
        cal[tau] = (e.copy(), sxy, sy)
        rows.append(dict(chi=chi, syz=syz, tau=tau, kind="calibrated", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, syz_applied=sy,
                         **{f"meas_s{n}": v for n, v in zip(NAMES, s)}, iterations=it, **_crit(g)))
    P = powers(syz)
    corr = lambda t: np.concatenate([cal[t][0] - LIN * t * s_, [cal[t][1] - t * c, cal[t][2] - syz]])
    M = np.array([[t ** p for p in P] for t in taus])
    q = np.linalg.lstsq(M, np.array([corr(t) for t in taus]), rcond=None)[0]
    entry = {f"q{p}": q[i].tolist() for i, p in enumerate(P)}
    for tau in taus_check:
        d = sum(q[i] * tau ** p for i, p in enumerate(P))
        e, sxy, sy = LIN * tau * s_ + d[:3], tau * c + d[3], syz + d[4]
        s = measure(f"yz_chi{chi:+g}_syz{syz:+g}_tau{tau}_model", e, sxy, sy)
        g = gate_a3_check(s, chi, tau, syz)
        log(f"chi {chi:+g} syz {syz:+g} tau {tau} MODEL: {fmt(s)} | ok {g['ok']}")
        rows.append(dict(chi=chi, syz=syz, tau=tau, kind="model_check", exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy, syz_applied=sy,
                         **{f"meas_s{n}": v for n, v in zip(NAMES, s)}, iterations=0, **_crit(g)))
    return rows, entry, all(r["gate_A3"] == "PASS" for r in rows)


def _crit(g):
    return dict(err_xy_pct=g["rxy"] * 100, err_xz_pct=g["rxz"] * 100, err_xz_angle_deg=np.degrees(g["ang"]), err_yz_pct=g["ryz"] * 100,
                err_yz_abs=g["yz_abs"], floor=g["floor"], other_pct=g["oth"] * 100, gate_A3="PASS" if g["ok"] else "FAIL")


def lammps_measure(np_ranks, out_dir="."):
    """Run from checks/10_gateA_yz (relative inputs). out_dir: where the LAMMPS logs and the temporary rebox data files go
    ("." = this folder, the step-3 behaviour; production passes its own folder, V02_PLAN 8.12 D-v02-10 log-file note)."""
    def lmp(inp, log, **v):
        args = ["mpirun", "-np", str(np_ranks), "lmp", "-in", inp, "-log", log, "-screen", "none"]
        for k, x in v.items():
            args += ["-var", k, str(x)]
        subprocess.run(args, check=True)
        return open(log).read()

    sys.path.insert(0, os.path.join(HERE, "../../tools"))
    from rebox import rebox
    if not os.path.exists("slab0.data"):
        lmp("../08_gateA_rebox/in.build", "log.build")

    def raw(tag, e, sxy, syz):
        t = open("slab0.data").read()
        Lx0 = float(re.search(r"^0 (\S+) xlo xhi", t, re.M).group(1)); Lz0 = float(re.search(r"^0 (\S+) zlo zhi", t, re.M).group(1))
        Lz = Lz0 * (1 + e[1])
        dat = os.path.join(out_dir, f"{tag}.data")
        rebox("slab0.data", dat, Lx0 * (1 + e[0]), Lz, e[2] * Lz)
        out = lmp("in.measure_yz", os.path.join(out_dir, f"log.{tag}"), tag=tag, datain=dat, exx=f"{e[0]:.10f}", ezz=f"{e[1]:.10f}",
                  sxy=f"{sxy:.10f}", syz=f"{syz:.10f}")
        os.remove(dat)
        d = dict(kv.split("=") for kv in re.search(r"^INTERIOR_GPa \S+ (.*)$", out, re.M).group(1).split())
        return np.array([float(d["s" + n]) for n in NAMES])

    ref = raw("yz_ref", np.zeros(3), 0.0, 0.0)
    return lambda tag, e, sxy, syz: raw(tag, e, sxy, syz) - ref


if __name__ == "__main__":
    chi, syz = float(sys.argv[1]), float(sys.argv[2])
    rows, entry, passed = calibrate(chi, syz, lammps_measure(os.environ.get("NP", "7")), slab_stiffness())
    new = not os.path.exists("calibration_yz.csv")
    with open("calibration_yz.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    if passed:
        m = json.load(open("strain_model_yz.json")) if os.path.exists("strain_model_yz.json") else {"correction": {}}
        m["correction"][f"{chi:g},{syz:+.2f}"] = entry
        m["note"] = "v0.2 Gate A-yz entries (calibrate_yz.py): 5 components (eps_xx, eps_zz, gamma_xz, sigma_xy and sigma_yz surface stress)"
        json.dump(m, open("strain_model_yz.json", "w"), indent=1)
    n_ok = sum(r["gate_A3"] == "PASS" for r in rows)
    print(f"GATE_A3 chi {chi:+g} syz {syz:+g}: {'PASS' if passed else 'FAIL'} ({n_ok}/{len(rows)})")
    sys.exit(0 if passed else 1)
