"""Shared ramp machinery for steps 5 and 6 (V02_PLAN section 5.4 rules 3-4, section 8.11 D-v02-6; 3 Nov 2026).

One ramp run (chi, sigma_yz, cell), each state relaxed from the previous one through checks/11_core_gate/run_step4.LammpsRunner
(tools/in.step_chi via tools/loading_v02.step_cmd; v0.1 sweep minimiser settings):
  start: tau 0, sigma_yz 0 (entry "<chi>,+0.00"), its DD state is the reference for the D5 rule;
  preyz: sigma_yz in 0.25 steps up to the target at tau 0, each level with its own step-3 entry (as step 4, D-v02-4 item 1);
  coarse: tau = 0.05, 0.10, ... 3.00 GPa (the tools/peierls_chi.py --coarse 0:3.0:0.05 schedule) until the first move;
  fine: tau = last unmoved coarse + 0.01 j (j = 1..4), from the last unmoved coarse state, until the first move
        (the peierls_chi.py refinement; bracket width 0.01 GPa);
  stab1, stab2: tools/stability_test.py at the last unmoved state (seeds 1, 2, v0.1 settings) via LammpsRunner.stability
        (a non-zero helper exit raises: a crash, never a failed check).
"Moved" = D5 (tools/moved_rule.classify relative to the start state): glide jump or easy -> split. An 'anomaly' (a compact core
shifted by less than one jump, moved_rule L11) is logged, not moved.
Rule-3 checks of a run: bracket found; first-moved tau <= 3.0 - 0.05 = 2.95 GPa; the first event (first state, along the ramp
path, whose D5 class is not 'none') is a glide jump or easy -> split, not an anomaly; no move during the sigma_yz pre-ramp;
both stability seeds return.
Checkpoint: one CSV line per state; resume replays the schedule from the CSV (the schedule depends only on the recorded
outcomes). Every possible LAMMPS input of a run (all coarse and 0.01-grid tau up to 3.0) is rendered and parsed before
running, and every possible stability argv is parsed with stability_test.py's parser.
"""
import csv, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
for p in ("tools", "checks/11_core_gate"):
    sys.path.insert(0, os.path.join(ROOT, p))
import run_step4 as r4
from moved_rule import classify
from loading_v02 import step_cmd
from render_lmp import render, cmd_vars, bad_numbers

TOP, COARSE, FINE, LADDER = 3.0, 0.05, 0.01, 0.25
TAU_MAX_BRACKET = round(TOP - COARSE, 2)                    # rule 3: bracket below the ramp top minus one coarse step
COARSE_TAUS = [round(COARSE * k, 2) for k in range(1, int(round(TOP / COARSE)) + 1)]
CELL_1X = dict(cell=r4.CELL, npy=r4.NPY, nz=21, z0=72.2716)
CELL_2X = dict(cell=os.path.join(ROOT, "sims/03_peierls_chi0_2x/relaxed_down.data"),
               npy=[os.path.join(ROOT, f"sims/03_peierls_chi0_2x/screw_down.data.{k}.npy") for k in ("ref", "core")], nz=42, z0=146.8746)
FIELDS = ["run", "idx", "phase", "chi", "syz", "tau", "model_key", "dd_dy", "dd_dz", "orient", "topcol_b", "event", "moved",
          "stab_returned", "pe_eV", "fmax", "fnorm", "stop", "seconds", "data"]


def run_name(chi, syz):
    return f"chi{chi:+d}_syz{syz:+.2f}"


def preyz_levels(syz):
    n = int(round(abs(syz) / LADDER))
    return [round(np.sign(syz) * LADDER * k, 2) for k in range(1, n + 1)]


def key(run, phase, syz, tau):
    return (run, phase, f"{syz:+.2f}", f"{tau:.2f}")


def done_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row); f.flush(); os.fsync(f.fileno())


def run_ramp(chi, syz, runner, model, results, work, say=print, rn=None, levels=None, entry_fn=None):
    """One ramp run; returns its summary (see summarise). Production (4 Nov, V02_PLAN 8.12) passes its own run name rn,
    pre-ramp levels (note a) and entry lookup entry_fn(model, chi, syz) (note b); the defaults are the steps 5-6 behaviour."""
    rn = rn or run_name(chi, syz); rdir = os.path.join(work, rn); runner.setup(rdir)
    levels = preyz_levels(syz) if levels is None else levels
    entry_fn = entry_fn or r4.entry_for
    done = {key(r["run"], r["phase"], float(r["syz"]), float(r["tau"])): r for r in done_rows(results) if r["run"] == rn}
    t0, idx, dd0, rows = time.time(), [0], None, []

    def state(phase, s, tau, datain):
        nonlocal dd0
        k = key(rn, phase, s, tau)
        if k in done:
            r = done[k]
        else:
            mk = r4.model_key(chi, s)
            ts = time.time()
            res = runner.relax(rdir, f"{idx[0]:02d}_{phase}_syz{s:+.2f}_tau{tau:.2f}", datain, entry_fn(model, chi, s), chi, tau, s)
            d = res["dd"]
            if phase == "start":
                dd0 = d
            ev = classify(d["dy"] - dd0["dy"], d["dz"] - dd0["dz"], d["orient"], d["topcol"])
            r = dict(run=rn, idx=idx[0], phase=phase, chi=chi, syz=s, tau=tau, model_key=mk, dd_dy=d["dy"], dd_dz=d["dz"], orient=d["orient"],
                     topcol_b=d["topcol"], event=ev, moved="yes" if ev in ("glide_jump", "split") else "no", pe_eV=res["pe"],
                     fmax=res["fmax"], fnorm=res.get("fnorm", ""), stop=res["stop"], seconds=round(time.time() - ts, 1), data=res["data"])
            append(results, r)
            say(f"{rn} state {idx[0] + 1} {phase} syz {s:+.2f} tau {tau:.2f} | {ev} | elapsed {time.time() - t0:.0f} s")
        if phase == "start" and dd0 is None:
            dd0 = dict(dy=float(r["dd_dy"]), dz=float(r["dd_dz"]))
        idx[0] += 1; rows.append(r)
        return r

    cur = state("start", 0.0, 0.0, runner.cell)
    path = [cur]                                           # the ramp path in tau order, up to the first move
    pre_moved = False
    for lv in levels:
        cur = state("preyz", lv, 0.0, cur["data"]); path.append(cur)
        if cur["moved"] == "yes":
            pre_moved = True; break
    lu, fm = None, None
    if not pre_moved:
        lu = cur
        for t in COARSE_TAUS:
            r = state("coarse", syz, t, lu["data"])
            if r["moved"] == "yes":
                fm = r; break
            lu = r; path.append(r)
        if fm is not None:
            coarse_fm, base, prev = fm, float(lu["tau"]), lu
            for j in range(1, int(round(COARSE / FINE))):
                r = state("fine", syz, round(base + FINE * j, 2), prev["data"]); path.append(r)
                if r["moved"] == "yes":
                    fm = r; break
                prev = lu = r
            if fm is coarse_fm:                            # no fine move: the coarse step is the first move on the path
                path.append(coarse_fm)
    stabs = []
    if lu is not None and fm is not None:
        for seed in (1, 2):
            k = key(rn, f"stab{seed}", syz, float(lu["tau"]))
            if k in done:
                r = done[k]
            else:
                ts = time.time()
                ok, line = runner.stability(rdir, f"{rn}_stab{seed}", lu["data"], r4.model_key(chi, syz), chi, float(lu["tau"]), syz, seed)
                r = dict(run=rn, idx=idx[0], phase=f"stab{seed}", chi=chi, syz=syz, tau=float(lu["tau"]), model_key=r4.model_key(chi, syz),
                         stab_returned="yes" if ok else "no", stop=line[:120], seconds=round(time.time() - ts, 1), data=lu["data"])
                append(results, r)
                say(f"{rn} state {idx[0] + 1} stab{seed} at tau {float(lu['tau']):.2f} | returned {r['stab_returned']} | elapsed {time.time() - t0:.0f} s")
            idx[0] += 1; stabs.append(r["stab_returned"] == "yes")
    return summarise(rn, chi, syz, path, lu, fm, pre_moved, stabs)


def summarise(rn, chi, syz, path, lu, fm, pre_moved, stabs):
    """path: the relaxed states along the ramp in tau order up to (and including) the first move."""
    events = [r["event"] for r in path if r["event"] != "none"]
    first = events[0] if events else None
    mid = (float(lu["tau"]) + float(fm["tau"])) / 2 if (lu is not None and fm is not None) else None
    checks = dict(bracket=mid is not None, below_top=fm is not None and float(fm["tau"]) <= TAU_MAX_BRACKET + 1e-9,
                  first_event_ok=first in ("glide_jump", "split"), no_preramp_move=not pre_moved,
                  stability=len(stabs) == 2 and all(stabs))
    return dict(run=rn, chi=chi, syz=syz, last_unmoved=None if lu is None else float(lu["tau"]),
                first_moved=None if fm is None else float(fm["tau"]), mid=mid, event=None if fm is None else fm["event"],
                first_event=first, checks=checks, rule3_pass=all(checks.values()))


def cell_consts(cell):
    ref = open(cell).read().splitlines()
    g = lambda k: [float(v) for v in next(l for l in ref[:30] if l.rstrip().endswith(k)).split()[:2]]
    LX0, LZ0 = g("xlo xhi")[1] - g("xlo xhi")[0], g("zlo zhi")[1] - g("zlo zhi")[0]
    XZ0 = float(next(l for l in ref[:30] if l.rstrip().endswith("xy xz yz")).split()[1])
    c = r4.read_cell(cell); y = c["r"][:, 1]
    return LX0, LZ0, XZ0, int((y > y.max() - 1.5 * 2.8553 * np.sqrt(2) / 2).sum())


def pre_run_checks(specs, model, entry_fn=None, model_path=None):
    """Render + parse every possible LAMMPS input (all tau on the 0.01 grid up to 3.0, pre-ramp levels, start) and parse every
    possible stability argv, for runs [(chi, syz, cellspec)] or, in production (4 Nov), [(chi, syz, cellspec, run name, levels)].
    Returns (n_inputs, n_argv, problems)."""
    tmpl, n, na, problems = os.path.join(ROOT, "tools/in.step_chi"), 0, 0, []
    grid = [round(FINE * k, 2) for k in range(0, int(round(TOP / FINE)) + 1)]
    entry_fn = entry_fn or r4.entry_for
    for spec in specs:
        chi, syz, cs = spec[:3]
        rn, levels = (spec[3], spec[4]) if len(spec) > 3 else (run_name(chi, syz), preyz_levels(syz))
        LX0, LZ0, XZ0, n_slab = cell_consts(cs["cell"])
        todo = [(0.0, 0.0)] + [(lv, 0.0) for lv in levels] + [(syz, t) for t in grid if t > 0]
        for s, t in todo:
            try:
                entry = entry_fn(model, chi, s)
            except KeyError:
                problems.append(f"{rn}: missing strain-model entry {r4.model_key(chi, s)}"); continue
            Lx, Lz, xz, fx, fz, e = r4.step_numbers(entry, chi, t, s, LX0, LZ0, XZ0, n_slab)
            cmd = step_cmd(7, tmpl, "log", "P", "in.data", "o.data", "o.pe", fx, r4.Y0, Lx, Lz, xz, r4.Y0, cs["z0"] * (1 + e[1]),
                           r4.MIN["maxit"], "g.lmp", 3 * r4.JUMP, r4.MIN["etol"], r4.MIN["ftol"], r4.MIN["dmax"], r4.MIN["maxeval"], fz=fz)
            problems += [f"{rn} tau {t:.2f} syz {s:+.2f}: {b}" for b in bad_numbers(render(tmpl, cmd_vars(cmd)), cmd)]
            n += 1
            if s == syz:
                for seed in (1, 2):
                    argv = r4.stab_argv("x.data", t, chi, f"{rn}_stab{seed}", seed, 7, r4.model_key(chi, syz), syz, cs["nz"], cs["cell"],
                                        model_path or r4.MODEL)
                    try:
                        a = r4.stability_parser().parse_args(argv)
                        if a.model_key != r4.model_key(chi, syz) or a.syz != syz or a.chi != chi or a.model != (model_path or r4.MODEL):
                            problems.append(f"{rn}: stability argv parsed wrongly")
                    except SystemExit:
                        problems.append(f"{rn}: stability argv rejected {argv}")
                    na += 1
    return n, na, problems


def max_states(syz, levels=None):
    return 1 + len(preyz_levels(syz) if levels is None else levels) + len(COARSE_TAUS) + int(round(COARSE / FINE)) - 1 + 2
