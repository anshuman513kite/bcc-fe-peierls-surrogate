"""Job lists for night T and night P (V02_PLAN section 8.12, D-v02-8; 4 Nov 2026). No LAMMPS.

T: the 7 x 7 held-out grid of section 4.1 (src/surrogate/designs.py grid_T: chi -30 ... +30 step 10, sigma_yz 0, +-s/3,
   +-2s/3, +-s) in GPa at s = 0.75. The 9 nodes already computed in step 5 (3 phase-A runs at sigma_yz 0 and 6 corners at
   +-s; D-v02-8 item 2) are imported from checks/12_corner_gate/step5_summary_s0.75.json into the sealed T results with a
   provenance column.
P: make_pool(376) exactly as in the synthetic study (item 3), sigma in GPa = level x s, computed in the fixed permutation
   numpy.random.default_rng(PERM_SEED).permutation(376) (item 4).
Pre-ramp levels (implementation note a): the same-sign sigma levels of the point's own set up to its target, so every level
is a pair of that set with its own calibration.
Asserts: T and P are disjoint (no shared pair, no shared sigma level); strain-model keys (run_step4.model_key) unique over
T and P; every pre-ramp step <= 0.25 GPa.
Writes data/production/T_jobs.csv and P_jobs.csv; imports the 9 step-5 nodes into data/sealed/T_results.csv (idempotent).
Usage (lammps env, from the project root): python tools/make_jobs_v02.py
"""
import csv, importlib.util, json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "checks/11_core_gate"))
from run_step4 import model_key
# designs.py is loaded from its file: the surrogate package __init__ imports torch, which the lammps env does not have
_spec = importlib.util.spec_from_file_location("designs_v02", os.path.join(ROOT, "src/surrogate/designs.py"))
_designs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_designs)
grid_T, make_pool = _designs.grid_T, _designs.make_pool

S, P_SIZE, PERM_SEED, MAX_STEP = 0.75, 376, 20261104, 0.25
PROD = os.path.join(ROOT, "data/production")
SEALED_T = os.path.join(ROOT, "data/sealed/T_results.csv")             # read only by the comparison script (item 1)
STEP5_SUMMARY = os.path.join(ROOT, "checks/12_corner_gate/step5_summary_s0.75.json")
STEP5_COMMIT = "b1bec26"
JOB_FIELDS = ["id", "set", "pool_index", "perm_rank", "chi", "syz_unit", "syz", "model_key", "levels", "provenance"]
RESULT_FIELDS = ["id", "set", "chi", "syz", "model_key", "last_unmoved", "first_moved", "tau_P_mid", "event", "first_event", "valid",
                 "reason", "below_top", "provenance", "source", "night", "n_new_calib", "calib_seconds", "ramp_seconds", "finished"]


def gpa(u):
    return round(float(u) * S, 10) + 0.0                                 # +0.0: -0.0 -> 0.0


def levels_for(syz, set_levels):
    """Pre-ramp levels (note a): same-sign levels of the set with |level| <= |syz|, in increasing |level|."""
    if syz == 0:
        return []
    lv = sorted({v for v in set_levels if v != 0 and np.sign(v) == np.sign(syz) and abs(v) <= abs(syz) + 1e-12}, key=abs)
    assert lv and abs(lv[-1] - syz) < 1e-12, (syz, lv)
    return lv


def build():
    """Returns (T jobs, P jobs) as lists of dicts (JOB_FIELDS); asserts the section 8.12 invariants."""
    T = grid_T()
    t_lv = sorted({gpa(u) for u in T[:, 1]})
    tj = []
    for i, (c, u) in enumerate(T):
        syz = gpa(u)
        tj.append(dict(id=f"T{i:02d}", set="T", pool_index="", perm_rank="", chi=float(c), syz_unit=float(u), syz=syz,
                       model_key=model_key(float(c), syz), levels=levels_for(syz, t_lv), provenance="night T"))
    P, nc, ns = make_pool(P_SIZE)
    assert len(P) == P_SIZE and (nc, ns) == (45, 8)
    perm = np.random.default_rng(PERM_SEED).permutation(P_SIZE)
    rank = {int(k): r for r, k in enumerate(perm)}
    p_lv = sorted({gpa(u) for u in P[:, 1]})
    pj = []
    for k, (c, u) in enumerate(P):
        syz = gpa(u)
        pj.append(dict(id=f"P{k:03d}", set="P", pool_index=k, perm_rank=rank[k], chi=float(c), syz_unit=float(u), syz=syz,
                       model_key=model_key(float(c), syz), levels=levels_for(syz, p_lv), provenance="night P"))
    pj.sort(key=lambda j: j["perm_rank"])
    # invariants (section 8.12 items 3, 4; note a)
    tp = {(round(j["chi"], 9), j["syz"]) for j in tj}
    assert not tp & {(round(j["chi"], 9), j["syz"]) for j in pj}, "T and P share a pair"
    assert not set(t_lv) & set(p_lv), "T and P share a sigma level"
    keys = [j["model_key"] for j in tj + pj]
    assert len(keys) == len(set(keys)), "strain-model keys not unique over T and P"
    for j in tj + pj:
        steps = np.diff([0.0] + [abs(v) for v in j["levels"]])
        assert all(0 < d <= MAX_STEP + 1e-12 for d in steps), (j["id"], j["levels"])
    return tj, pj


def step5_nodes(tj, path=STEP5_SUMMARY):
    """The 9 T nodes computed in step 5 (item 2), as RESULT_FIELDS rows."""
    d = json.load(open(path))
    src = [(x, "step5 phase A") for x in d["phaseA"].values()] + [(x, "step5 corner") for x in d["corners"].values()]
    by_pair = {(round(j["chi"], 9), j["syz"]): j for j in tj}
    rows = []
    for x, prov in src:
        j = by_pair[(round(float(x["chi"]), 9), float(x["syz"]) + 0.0)]
        bad = [k for k, ok in x["checks"].items() if not ok and k != "below_top"]
        rows.append(dict(id=j["id"], set="T", chi=j["chi"], syz=j["syz"], model_key=j["model_key"], last_unmoved=x["last_unmoved"],
                         first_moved=x["first_moved"], tau_P_mid=x["mid"], event=x["event"], first_event=x["first_event"],
                         valid="yes" if not bad else "no", reason=";".join(bad), below_top="yes" if x["checks"]["below_top"] else "no",
                         provenance=prov, source=f"checks/12_corner_gate/step5_summary_s0.75.json ({x['run']}, commit {STEP5_COMMIT})",
                         night="", n_new_calib=0, calib_seconds="", ramp_seconds="", finished=""))
    assert len(rows) == 9 and len({r["id"] for r in rows}) == 9
    return rows


def import_step5(tj, sealed=SEALED_T, path=STEP5_SUMMARY):
    """Append the 9 step-5 nodes to the sealed T results unless already there. Returns the number added."""
    have = {r["id"] for r in csv.DictReader(open(sealed))} if os.path.exists(sealed) else set()
    new = [r for r in step5_nodes(tj, path) if r["id"] not in have]
    os.makedirs(os.path.dirname(sealed), exist_ok=True)
    first = not os.path.exists(sealed)
    with open(sealed, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if first:
            w.writeheader()
        w.writerows(new)
    return len(new)


def write_jobs(jobs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader()
        w.writerows([dict(j, levels=";".join(f"{v:+.5f}" for v in j["levels"])) for j in jobs])


def main():
    tj, pj = build()
    imported = {r["id"]: r["provenance"] for r in step5_nodes(tj)}
    for j in tj:
        j["provenance"] = imported.get(j["id"], j["provenance"])
    write_jobs(tj, os.path.join(PROD, "T_jobs.csv")); write_jobs(pj, os.path.join(PROD, "P_jobs.csv"))
    n = import_step5(tj)
    print(f"T: {len(tj)} points ({len(imported)} done in step 5, {n} newly imported into {os.path.relpath(SEALED_T, ROOT)}); "
          f"P: {len(pj)} points, permutation seed {PERM_SEED}, first 5 in order {[j['id'] for j in pj[:5]]}; disjoint: yes")


if __name__ == "__main__":
    main()
