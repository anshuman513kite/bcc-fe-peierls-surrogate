import csv, json, os, re, sys, glob
import numpy as np
ROOT = os.getcwd()
for p in ("tools", "checks/11_core_gate", "checks/12_corner_gate", "checks/14_production", "checks/10_gateA_yz"):
    sys.path.insert(0, os.path.join(ROOT, p))
import run_step4 as r4, ramp_v02 as rp, run_production as pr
from calibrate_yz import LIN
night = sys.argv[1]
ids = json.load(open(f"data/production/P_night{night}.json"))
ids = ids["ids"] if isinstance(ids, dict) else ids
model = json.load(open(pr.PROD_MODEL))["correction"]
runner = r4.LammpsRunner("7", **rp.CELL_1X, model_path=pr.PROD_MODEL)
LX0, LZ0, XZ0, ns = runner.LX0, runner.LZ0, runner.XZ0, len(runner.top)
res = {r["id"]: r for r in csv.DictReader(open("data/production/P_results.csv"))}
states = [r for r in csv.DictReader(open("data/production/P_states.csv"))]
bad, nlog, zero_starts, pts_ok = [], 0, 0, 0
for pid in ids:
    run = [d for d in os.listdir("sims/12_production/P") if d.startswith(pid + "_")]
    assert len(run) == 1, pid
    run = run[0]; rdir = f"sims/12_production/P/{run}"
    chi = float(res[pid]["chi"]); tsyz = float(res[pid]["syz"])
    efn = pr.entry_fn_for(tsyz)
    rows = [r for r in states if r["run"] == run]
    ok = True
    for r in rows:
        syz, tau, ph = float(r["syz"]), float(r["tau"]), r["phase"]
        k = r4.model_key(chi, syz)
        if r["model_key"] != k: bad.append((pid, ph, "row key", r["model_key"], k)); ok = False
        if k not in model:
            if syz == 0 and ph == "start": zero_starts += 1
            else: bad.append((pid, ph, "no entry", k)); ok = False
        entry = efn({"correction": model}, chi, syz); entry = {kk: [v * 1.001 for v in vv] for kk, vv in entry.items()}
        Lx, Lz, xz, fx, fz, e = r4.step_numbers(entry, chi, tau, syz, LX0, LZ0, XZ0, ns)
        if ph.startswith("stab"):
            log = f"{rdir}/stab/log.{run}_{ph}"
        else:
            log = f"{rdir}/steps/log.{int(r['idx']):02d}_{ph}_syz{syz:+.2f}_tau{tau:.2f}"
        if not os.path.exists(log): bad.append((pid, ph, "log missing", log)); ok = False; continue
        t = open(log).read(); nlog += 1
        m = re.findall(r"^fix\s+ftop top addforce (\S+) 0\.0 (\S+)$", t, re.M)
        fxl, fzl = m[-1]
        exp_fz = f"{fz:.10f}" if fz is not None and fz != 0.0 else None
        if fxl != f"{fx:.10f}": bad.append((pid, ph, tau, "fx", fxl, f"{fx:.10f}")); ok = False
        if exp_fz is None:
            if fzl not in ("0.0", "${fz}"): bad.append((pid, ph, tau, "fz", fzl, "0.0")); ok = False
        elif fzl != exp_fz: bad.append((pid, ph, tau, "fz", fzl, exp_fz)); ok = False
        b = re.search(r"triclinic box = \(0 \S+ 0\) to \((\S+) \S+ (\S+)\) with tilt \(0 (\S+) 0\)", t)
        bx = [float(v) for v in b.groups()]
        for nm, got, want in (("Lx", bx[0], Lx), ("Lz", bx[1], Lz), ("xz", bx[2], xz)):
            if abs(got - want) > 5e-6 * max(1.0, abs(want)): bad.append((pid, ph, tau, nm, got, want)); ok = False
    if not ok: pass
    pts_ok += ok
logs_on_disk = sum(len(glob.glob(f"sims/12_production/P/{[d for d in os.listdir('sims/12_production/P') if d.startswith(p + '_')][0]}/steps/log.*"))
                   + len(glob.glob(f"sims/12_production/P/{[d for d in os.listdir('sims/12_production/P') if d.startswith(p + '_')][0]}/stab/log.*")) for p in ids)
# calibrations
cal = [r for r in csv.DictReader(open(pr.CALIB_CSV)) if r["for_point"] in set(ids)]
cal_ok, nml = 0, 0
for c in cal:
    k, chi, syz = c["model_key"], float(c["chi"]), float(c["syz"])
    q = model.get(k); okc = q is not None and c["verdict"] == "PASS"
    sn, cs = np.sin(np.radians(chi)), np.cos(np.radians(chi))
    logs = sorted(glob.glob(f"checks/14_production/calib/{k.replace(',', '_')}/log.*_model"))
    if len(logs) != 7: bad.append((k, "model logs", len(logs))); okc = False
    for lg in logs:
        tau = float(re.search(r"_tau([0-9.]+)_model$", lg).group(1)); nml += 1
        d = sum(np.asarray(q[f"q{p}"]) * tau ** p for p in range(5) if f"q{p}" in q)
        e, sxy, sy = LIN * tau * sn + d[:3], tau * cs + d[3], syz + d[4]
        t = open(lg).read()
        info = dict(kv.split("=") for kv in re.search(r"^INFO \S+ (.*)$", t, re.M).group(1).split())
        top = int(info["top"])
        lx, lz = 24.727623354257076 * (1 + e[0]), 27.976112250275232 * (1 + e[1])
        m = re.findall(r"^fix\s+ftop top addforce (\S+) 0\.0 (\S+)$", t, re.M)[-1]
        fx, fz = sxy * lx * lz * 0.0062415091 / top, sy * lx * lz * 0.0062415091 / top
        chk = [("lx", float(info["lx"]), lx, 2e-6), ("lz", float(info["lz"]), lz, 2e-6), ("xz", float(info["xz"]), e[2] * lz, 2e-6),
               ("fx", float(m[0]), fx, 1e-9 * max(1, abs(fx)) + 1e-15), ("fz", float(m[1]), fz, 1e-9 * max(1, abs(fz)) + 1e-15)]
        for nm, got, want, tol in chk:
            if abs(got - want) > tol: bad.append((k, tau, nm, got, want)); okc = False
    cal_ok += okc
print(f"points {pts_ok}/{len(ids)}; state logs checked {nlog}/{logs_on_disk} on disk ({sum(1 for r in states if r['run'].split('_')[0] in set(ids))} state rows); zero-correction starts {zero_starts}")
print(f"calibrations {cal_ok}/{len(cal)}; model-check logs {nml}")
print("mismatches:", len(bad)); [print("  ", b) for b in bad[:15]]
