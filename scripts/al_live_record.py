"""Live active learning, record step (Session 12; rule D5 from Session 14): read the finished run(s) of one round and write
its result to data/al_live_results.csv, with the comparison against the pre-registered prediction (data/al_live_predictions.csv).

Every step is (re)classified with Ansh's D5 rule (tools/moved_rule.py), label-free, from the DD map of its step file
(tools/moved_rule.dd_state) relative to the tau = 0 state: moved = a glide jump (>= one jump to a compact 'down' site) or an
easy -> split transition (top column >= 1.4 b). Event type recorded. (Session 13 used D1, Session 12 the any-change rule.)
Several run names (comma-separated) are merged in tau order, e.g. a ramp and its continuation (chi = +28, Session 13).
tau_mid = midpoint of the jump bracket (last unmoved, first moved; first moved = the moved step with the LOWEST tau; the
Session 12 first version took the first moved row in file order, archived as data/al_live_results_recorderbug.csv).
Stability: the STABILITY lines of the run holding the last unmoved step; lines from the D5 check (with returned_triangle=)
are preferred over older ones. If the round is already in the results file, its
row is replaced (the pre-registered prediction is never touched).
Usage (lammps env): python scripts/al_live_record.py <round> <chi> <name>[,<name>...] <stability.txt>
"""
import csv, os, re, subprocess, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from moved_rule import classify, is_moved, dd_state, EVENT_NAME, JUMP
k, chi, names, stab = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
PLANE = {0: "(-110)", 60: "(0 1 -1)", -60: "(-1 0 1)"}


def state(name, r):
    return dd_state(f"steps/{r['kind']}_{float(r['tau_GPa']):.3f}.data", f"{ROOT}/sims/07_al_live/{name}", 21)


steps = []
for name in names:
    for r in csv.DictReader(open(os.path.join(ROOT, "data", f"{name}.csv"))):
        r["run"] = name; steps.append(r)
steps.sort(key=lambda r: float(r["tau_GPa"]))
s0 = next(r for r in steps if float(r["tau_GPa"]) == 0.0)
S0 = state(s0["run"], s0)
for r in steps:
    r["st"] = state(r["run"], r)
    r["dy"], r["dz"] = r["st"]["dy"] - S0["dy"], r["st"]["dz"] - S0["dz"]
    r["cls"] = classify(r["dy"], r["dz"], r["st"]["orient"], r["st"]["topcol"])
fm = next((r for r in steps if is_moved(r["cls"])), None)
if fm is None:
    sys.exit(f"round {k}: no move under D5 up to {float(steps[-1]['tau_GPa']):.3f} GPa; nothing recorded")
lu = [r for r in steps if float(r["tau_GPa"]) < float(fm["tau_GPa"]) and not is_moved(r["cls"])][-1]
ang = float(np.degrees(np.arctan2(fm["dy"], fm["dz"]))); shift = float(np.hypot(fm["dy"], fm["dz"]))
plane = PLANE.get(int(round(ang / 60.0)) * 60, f"angle {ang:+.0f}") if fm["cls"] == "glide_jump" else "- (split core on a vertex column)"
stl = [l for l in open(stab) if l.startswith(f"STABILITY {lu['run']}") and "_last" in l.split()[1]]
stl = [l for l in stl if "returned_triangle=" in l] or stl
pred = next(r for r in csv.DictReader(open(os.path.join(ROOT, "data", "al_live_predictions.csv")))
            if int(r["round"]) == k and not r.get("status", "").startswith("voided"))
assert int(pred["chi_pick"]) == chi, "run angle differs from the pre-registered pick"
tl, tf = float(lu["tau_GPa"]), float(fm["tau_GPa"]); mid = 0.5 * (tl + tf)
z = (mid - float(pred["pred_mean"])) / float(pred["pred_spread"])
below = [r for r in steps if float(r["tau_GPa"]) < tl and r["converged"] != "yes"]
row = dict(round=k, chi_deg=chi, tau_last_unmoved=tl, tau_first_moved=tf, tau_mid=round(mid, 5), dd_shift_dy=round(fm["dy"], 4), dd_shift_dz=round(fm["dz"], 4),
           dd_shift_angle_deg=round(ang, 1), shift_A=round(shift, 4), jumps=round(shift / JUMP, 3), glide_plane=plane,
           event=EVENT_NAME[fm["cls"]], topcol_first_moved_b=fm["st"]["topcol"], topcol_last_unmoved_b=lu["st"]["topcol"],
           stability="; ".join(re.search(r" returned=(\S+)", l).group(1) + " (fmax " + re.search(r"fmax=(\S+)", l).group(1) + ")" for l in stl),
           nonconverged_below=",".join(r["tau_GPa"] for r in below), pred_mean=pred["pred_mean"], pred_spread=pred["pred_spread"], z=round(z, 2),
           inside_2sigma="yes" if abs(z) <= 2 else "no", rule="D5", source=";".join(f"data/{n}.csv" for n in names))
path = os.path.join(ROOT, "data", "al_live_results.csv")
old = [r for r in csv.DictReader(open(path)) if int(r["round"]) != k] if os.path.exists(path) else []
allrows = sorted(old + [row], key=lambda r: int(r["round"]))
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row), restval="", extrasaction="ignore"); w.writeheader(); w.writerows(allrows)   # retired columns of old rows dropped
print(f"RECORDED round {k}: chi {chi:+d}: bracket {tl:.3f}-{tf:.3f} GPa (mid {mid:.4f}), {row['event']} (DD shift {shift:.3f} A, "
      f"top column {fm['st']['topcol']:.3f} b, plane {plane}); predicted {float(pred['pred_mean']):.4f} +- {float(pred['pred_spread']):.4f}; "
      f"z = {z:+.2f} -> inside +-2: {row['inside_2sigma']}; stability {row['stability'] or 'none found'}")
