"""Audit of the first "moved" step of every Peierls ramp under Ansh's D5 rule (Session 14; Session 13 version: D1, first
moved step only, archived as data/audit_moves_d1.csv).

Every saved step of every ramp is re-analysed with the DD core finder (tools/moved_rule.dd_state, in parallel) and classified
with tools/moved_rule.classify relative to the start site: 'glide jump' (>= one jump to a compact 'down' site),
'easy -> split transition' (top column >= 1.4 b), 'anomaly' (compact, shifted < one jump) or nothing. The first step (lowest
tau) with a glide jump or split gives the bracket (with the highest unmoved tau below it). Runs: the 10 small-cell sweep
angles, chi = 0 (Gate B', history, and B2, the baseline from D6), the 2x cells at +-30, the live rounds (+28: ramp and its
continuation merged). Reported per run: bracket, event type, DD shift and angle, glide plane (for glide jumps), top column,
any anomalies; and the change against the D1 audit.
Usage (lammps env, project root): python scripts/audit_moves.py   -> prints the table, writes data/audit_moves.csv
"""
import csv, glob, os, re, sys, tempfile
from multiprocessing import Pool
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{ROOT}/tools")
from moved_rule import classify, is_moved, dd_state, EVENT_NAME, JUMP
SCR = os.environ.get("AUDIT_PNG") or tempfile.mkdtemp(prefix="audit_dd_")       # scratch folder for the discarded DD-map images
PLANE = {0: "(-110)", 60: "(0 1 -1)", -60: "(-1 0 1)", 180: "(-110)", 120: "(-1 0 1)", -120: "(0 1 -1)"}
S6, S7, S8 = f"{ROOT}/sims/06_chi_sweep", f"{ROOT}/sims/07_al_live", f"{ROOT}/sims/08_block2"
RUNS = [(f"chi_small_{t}", "small", [f"{S6}/chi_small_{t}"]) for t in ("m30", "m24", "m18", "m12", "m6", "p6", "p12", "p18", "p24", "p30")]
RUNS += [("chi_small_0 (B2, baseline)", "small", [f"{S8}/chi_small_0"]), ("gateBp_small_np7 (chi = 0, history)", "small", [f"{ROOT}/sims/05_gate_bprime/ramp_small_np7"])]
RUNS += [(f"chi_x2_{t}", "2x", [f"{S6}/chi_x2_{t}"]) for t in ("m30", "p30")]
RUNS += [("chi_small_m28", "small (live 1)", [f"{S7}/chi_small_m28"]), ("chi_small_p28 (+cont)", "small (live 2)", [f"{S7}/chi_small_p28", f"{S7}/chi_small_p28_cont"])]
RUNS += [(os.path.basename(f), "small (live 3)", [f]) for f in sorted(glob.glob(f"{S7}/chi_small_m29"))]     # live round 3 (Session 14)


def job(a):
    folder, f, nz = a
    return folder, f, dd_state(f, folder, nz, os.path.join(SCR, f"_audit_{os.getpid()}.png"))


def steps_of(folder):
    out = []
    for f in glob.glob(f"{folder}/steps/*.data"):
        if f.endswith(".in.data"):
            continue
        m = re.search(r"/(coarse|fine)_(\d+\.\d+)\.data$", f)
        out.append((float(m.group(2)), m.group(1), os.path.relpath(f, folder)))
    return out


if __name__ == "__main__":
    jobs, meta = [], {}
    for name, cell, folders in RUNS:
        nz = 42 if cell == "2x" else 21
        for folder in folders:
            for tau, kind, f in steps_of(folder):
                jobs.append((folder, f, nz)); meta[(folder, f)] = (name, tau, kind)
    with Pool(6) as p:
        res = p.map(job, jobs)
    by = {}
    for folder, f, st in res:
        name, tau, kind = meta[(folder, f)]
        by.setdefault(name, []).append(dict(tau=tau, kind=kind, **st))
    old = {r["run"].split(" ")[0]: r for r in csv.DictReader(open(f"{ROOT}/data/audit_moves_d1.csv"))}
    rows = []
    for name, cell, folders in RUNS:
        st = sorted(by[name], key=lambda s: (s["tau"], s["kind"] == "coarse"))
        s0 = st[0]
        for s in st:
            s["sdy"], s["sdz"] = s["dy"] - s0["dy"], s["dz"] - s0["dz"]
            s["ev"] = classify(s["sdy"], s["sdz"], s["orient"], s["topcol"])
        mv = [s for s in st if is_moved(s["ev"])]
        fm = mv[0]
        lu = [s for s in st if s["tau"] < fm["tau"] and not is_moved(s["ev"])][-1]
        later_unmoved = [s["tau"] for s in st if s["tau"] > fm["tau"] and not is_moved(s["ev"])]
        anom = [s["tau"] for s in st if s["ev"] == "anomaly"]
        shift = float(np.hypot(fm["sdy"], fm["sdz"])); ang = float(np.degrees(np.arctan2(fm["sdy"], fm["sdz"])))
        plane = PLANE.get(int(round(ang / 60.0)) * 60, "?") if fm["ev"] == "glide_jump" else "- (split, on a vertex column)"
        key = name.split(" ")[0]
        o = old.get(key) or old.get({"chi_small_p28": "chi_small_p28"}.get(key, key))
        old_br = f"{float(o['tau_last_unmoved']):.3f}-{float(o['tau_first_moved']):.3f}" if o else "-"
        br = f"{lu['tau']:.3f}-{fm['tau']:.3f}"
        rows.append(dict(run=name, cell=cell, chi_deg=float(name.split("_")[-1].split(" ")[0].replace("m", "-").replace("p", "+")) if "chi_small_0" not in name and "gateBp" not in name else 0.0,
                         tau_last_unmoved=lu["tau"], tau_first_moved=fm["tau"], first_moved_kind=fm["kind"], event=EVENT_NAME[fm["ev"]],
                         shift_A=round(shift, 4), jumps=round(shift / JUMP, 3), angle_deg=round(ang, 1), orientation=fm["orient"], glide_plane=plane,
                         topcol_b=fm["topcol"], topcol_dydz=f"({fm['col'][0]:+.2f},{fm['col'][1]:+.2f})", topcol_last_unmoved_b=lu["topcol"],
                         anomalies=",".join(f"{x:.3f}" for x in anom), unmoved_after_first_move=",".join(f"{x:.3f}" for x in later_unmoved),
                         bracket_d1=old_br, changed_vs_d1="" if old_br in ("-", br) else "CHANGED"))
    print(f"{'run':36s} {'chi':>4s} {'D5 bracket':>12s} {'D1 bracket':>12s} {'event':25s} {'shift A':>7s} {'angle':>6s} {'plane':30s} {'top col b':>9s} {'(last unm.)':>11s} notes")
    for r in rows:
        notes = " ".join(x for x in (r["changed_vs_d1"], f"anomalies at {r['anomalies']}" if r["anomalies"] else "",
                                     f"unmoved again at {r['unmoved_after_first_move']}" if r["unmoved_after_first_move"] else "") if x)
        print(f"{r['run']:36s} {r['chi_deg']:+4.0f} {r['tau_last_unmoved']:.3f}-{r['tau_first_moved']:.3f} {r['bracket_d1']:>12s} {r['event']:25s} "
              f"{r['shift_A']:7.3f} {r['angle_deg']:+6.1f} {r['glide_plane']:30s} {r['topcol_b']:9.3f} {r['topcol_last_unmoved_b']:11.3f} {notes}")
    with open(f"{ROOT}/data/audit_moves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
