"""Post-hoc test of tools/track_core.py on the chi = 0 ramps (no new simulation; per-atom energies from tools/run_track.sh).

Each step is tracked from the core position of the configuration it started from (ramp order: coarse steps in sequence;
fine steps start from the last stable coarse step), exactly as the driver chains them. Positions are printed relative to the
inserted core (y0, z0), next to the old tracker's values from data/<name>.csv (old y relative to y0 as well).
Usage (project root): python checks/06_tracker/test_tracker.py
"""
import csv, re, subprocess, sys
import numpy as np
A = 2.8553; JUMP = A * np.sqrt(6) / 3
for sim, cell, name, nz in [("02_peierls_chi0", "01_screw_cell", "peierls_chi0", 21), ("03_peierls_chi0_2x", "03_peierls_chi0_2x", "peierls_chi0_2x", 42)]:
    d = f"sims/{sim}"; LZ = nz * A * np.sqrt(6)
    y0, z0 = np.load(f"sims/{cell}/screw_down.data.core.npy")
    rows = list(csv.DictReader(open(f"data/{name}.csv")))
    def track(pe, yp, zp):
        out = subprocess.run([sys.executable, "tools/track_core.py", f"{d}/track/{pe}.pe", str(yp), str(zp), str(LZ)],
                             capture_output=True, text=True, check=True).stdout
        m = re.search(r"y=(\S+) z=(\S+) iterations=(\d+) n_atoms=(\d+)", out)
        return float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    ys, zs, *_ = track("relaxed_down", y0, z0)
    print(f"== {name} (Lz = {LZ:.2f} A): sigma = 0 relaxed cell: new tracker y - y0 = {ys - y0:+.4f}, z - z0 = {zs - z0:+.4f} A")
    pos = {}; prev = (ys, zs); last_stable = None
    old_z0 = float(rows[0]["core_z_A"])
    for r in rows:
        tag = f"{r['kind']}_{float(r['sigma_GPa']):.3f}"
        start = prev if r["kind"] == "coarse" or tag not in pos and r["kind"] == "fine" and pos.get("_fine_started") else prev
        if r["kind"] == "fine" and "_fine_started" not in pos:
            start = pos[last_stable]; pos["_fine_started"] = True
        yn, zn, it, na = track(tag, *start)
        pos[tag] = (yn, zn)
        dzn = (zn - z0 + LZ / 2) % LZ - LZ / 2
        dzo = (float(r["core_z_A"]) - old_z0 + LZ / 2) % LZ - LZ / 2
        print(f"  {tag:14s} new: dy = {yn - y0:+.3f} dz = {dzn:+.3f} A ({dzn / JUMP:+.3f} jumps; {it} it, {na} atoms) | "
              f"old: dy = {float(r['core_y_A']) - y0:+.3f} dz = {dzo:+.3f} | new-old dz = {dzn - dzo:+.3f}")
        if abs(float(r["dz_step_A"])) < JUMP:
            last_stable = tag
        prev = (yn, zn)
