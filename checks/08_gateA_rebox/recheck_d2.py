"""Session 13, A3: re-check the stored live-AL calibration points (calibration_al.csv) with Ansh's D2 sigma_xz criterion
(tools/gate_a.py), no new runs. Adds err_xz_angle_deg, rule (S12 = the original 1 %-only criterion) and gate_A_D2 to the file.
Usage: python recheck_d2.py
"""
import csv, sys
import numpy as np
sys.path.insert(0, "../../tools")
from gate_a import gate_a_check
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
rows = list(csv.DictReader(open("calibration_al.csv")))
for r in rows:
    s = np.array([float(r["s" + n]) for n in names])
    err, rxy, rxz, ang, oth, ok = gate_a_check(s, float(r["chi"]), float(r["tau"]))
    r.setdefault("rule", "S12"); r["err_xz_angle_deg"] = f"{np.degrees(ang):.5f}"; r["gate_A_D2"] = "PASS" if ok else "FAIL"
    print(f"chi {float(r['chi']):+3.0f} tau {float(r['tau']):4} {r['kind']:11s} it {r['iterations']}: d_sxz {err[4]:+.5f} GPa = {rxz*100:.3f} % of target, "
          f"{np.degrees(ang):.4f} deg; xy {rxy*100:.3f} %, other {oth*100:.3f} % of tau | old rule {r['gate_A']} -> D2 {r['gate_A_D2']}")
keys = list(rows[0].keys())
with open("calibration_al.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for chi in sorted({r["chi"] for r in rows}, key=float):
    rr = [r for r in rows if r["chi"] == chi]
    print(f"chi {float(chi):+3.0f}: old rule {sum(r['gate_A'] == 'PASS' for r in rr)}/{len(rr)}, D2 {sum(r['gate_A_D2'] == 'PASS' for r in rr)}/{len(rr)}")
