"""Gate A spot re-check with the rebox path (Session 12, 28 Oct).

The original Gate A (checks/07_chi_calibration) set the slab strains with `change_box ... remap`, which also stretched y by
~1e-4 per call (Session 11). Here the same perfect slab is strained with tools/rebox.py (x, z affine, y untouched) and the
interior stress is re-measured for chi = -30, -12, +12, +30 at tau = 1.5 and 2.0 GPa with
  (a) the calibrated strains of calibration.csv (the Gate A points), and
  (b) the tau-polynomial model strains of strain_model.json (what the sweep's peierls_chi.py actually applied).
Pass criteria = Gate A: sigma_xy and sigma_xz within 1 % of target, every other component below 2 % of tau.
Also compared with the original change_box measurement of the same calibrated point.
Run (crash-safe): ../../tools/launch_safe.sh gateA_rebox python recheck.py
"""
import csv, json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, "../../tools")
from rebox import rebox
NP = os.environ.get("NP", "7")
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
CHIS, TAUS = (-30, -12, 12, 30), (1.5, 2.0)
LIN = np.array([0.0, 0.0027625, 0.0164349])


def lmp(inp, log, **v):
    args = ["mpirun", "-np", NP, "lmp", "-in", inp, "-log", log, "-screen", "none"]
    for k, x in v.items():
        args += ["-var", k, str(x)]
    subprocess.run(args, check=True)
    return open(log).read()


def box(path):
    t = open(path).read()
    lx = [float(s) for s in re.search(r"^(\S+) (\S+) xlo xhi", t, re.M).groups()]
    lz = [float(s) for s in re.search(r"^(\S+) (\S+) zlo zhi", t, re.M).groups()]
    return lx[1] - lx[0], lz[1] - lz[0]


def measure(tag, e, sxy):
    Lx0, Lz0 = box("slab0.data")
    Lx, Lz = Lx0 * (1 + e[0]), Lz0 * (1 + e[1])
    rebox("slab0.data", f"{tag}.data", Lx, Lz, e[2] * Lz)      # in.block: stretch first, then xz = gxz * lz (new lz)
    out = lmp("in.measure", f"log.{tag}", tag=tag, datain=f"{tag}.data", exx=f"{e[0]:.10f}", ezz=f"{e[1]:.10f}", sxy=f"{sxy:.10f}")
    d = dict(kv.split("=") for kv in re.search(r"^INTERIOR_GPa \S+ (.*)$", out, re.M).group(1).split())
    info = re.search(r"^INFO .*$", out, re.M).group(0)
    os.remove(f"{tag}.data")
    return np.array([float(d["s" + n]) for n in names]), info


def check(s, chi, tau):
    t = np.zeros(6); t[5] = tau * np.cos(np.radians(chi)); t[4] = tau * np.sin(np.radians(chi))
    err = s - t
    rxy = abs(err[5]) / abs(t[5]); rxz = abs(err[4]) / abs(t[4]); oth = max(abs(err[[0, 1, 2, 3]])) / tau
    return rxy, rxz, oth, (rxy < 0.01 and rxz < 0.01 and oth < 0.02)


print(re.search(r"^BUILD.*$", lmp("in.build", "log.build"), re.M).group(0), flush=True)
ref, info = measure("ref", np.zeros(3), 0.0)
print("REF", info, " ".join(f"s{n}={x:+.6f}" for n, x in zip(names, ref)), flush=True)
cal = {(int(r["chi"]), float(r["tau"])): r for r in csv.DictReader(open("../07_chi_calibration/calibration.csv")) if r["kind"] == "calibrated"}
model = json.load(open("../07_chi_calibration/strain_model.json"))["correction"]
rows = []
for chi in CHIS:
    for tau in TAUS:
        r0 = cal[(chi, tau)]
        q = model[str(chi)]
        c = np.array(q["q2"]) * tau ** 2 + np.array(q["q3"]) * tau ** 3 + np.array(q["q4"]) * tau ** 4
        variants = {"calibrated": (np.array([float(r0["exx"]), float(r0["ezz"]), float(r0["gxz"])]), float(r0["sxy_applied"])),
                    "model": (LIN * tau * np.sin(np.radians(chi)) + c[:3], tau * np.cos(np.radians(chi)) + c[3])}
        old = np.array([float(r0["s" + n]) for n in names])
        for kind, (e, sxy) in variants.items():
            tag = f"chi{chi:+d}_tau{tau}_{kind}"
            s, info = measure(tag, e, sxy)
            s = s - ref
            rxy, rxz, oth, ok = check(s, chi, tau)
            dmax = max(abs(s - old))
            print(f"{tag}: " + " ".join(f"s{n}={x:+.6f}" for n, x in zip(names, s)) +
                  f" | xy {rxy*100:.3f} % xz {rxz*100:.3f} % other {oth*100:.3f} % of tau | {'PASS' if ok else 'FAIL'}"
                  f" | max |rebox - change_box(calibrated)| = {dmax:.6f} GPa | {info.split(' ', 2)[2]}", flush=True)
            rows.append(dict(chi=chi, tau=tau, strains=kind, exx=e[0], ezz=e[1], gxz=e[2], sxy_applied=sxy,
                             **{f"s{n}": v for n, v in zip(names, s)}, err_xy_pct=rxy * 100, err_xz_pct=rxz * 100,
                             other_pct_tau=oth * 100, max_diff_vs_changebox_GPa=dmax, gate_A="PASS" if ok else "FAIL"))
with open("recheck.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("GATE_A_REBOX calibrated:", "PASS" if all(r["gate_A"] == "PASS" for r in rows if r["strains"] == "calibrated") else "FAIL",
      "; model:", "PASS" if all(r["gate_A"] == "PASS" for r in rows if r["strains"] == "model") else "FAIL")
