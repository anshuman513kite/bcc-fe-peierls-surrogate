"""v0.2 loading helpers (2 Nov 2026), shared by tools/peierls_chi.py and tools/stability_test.py; pure Python, no LAMMPS.

- surface_force(): force per slab atom (eV/A) for a surface-force stress (GPa) on a slab of n atoms in a box Lx x Lz (A).
  Same conversion as v0.1 (Session 5 hand check: 0.017991 eV/A per atom per GPa in the 146.87 A cell).
- syz_ladder(): the sigma_yz pre-ramp at tau = 0 (Ansh, V02_PLAN section 8 Q1): steps of at most 0.25 GPa up to the target.
- step_cmd(): the LAMMPS command of one ramp step (tools/in.step_chi). With fz = None (sigma_yz = 0) it is exactly the v0.1
  command; with sigma_yz != 0 it adds "-var fz <top> -var fzb <bottom>" (the bottom force is passed as its own string);
  with fx < 0 it adds "-var fxb <-fx>" (3 Nov 2026: in.step_chi's default '-${fx}' would read '--...' for fx < 0).
- model_correction(): correction from a strain-model entry. v0.1 entries have q2, q3, q4 (polynomial in tau from tau^2),
  each a 4-vector (d eps_xx, d eps_zz, d gamma_xz, d sigma_xy surface stress). v0.2 entries (checks/10_gateA_yz) have
  5-vectors (+ d sigma_yz surface stress) and, at sigma_yz != 0, also q0, q1: a constant and a tau-linear term, because
  tau * sigma_yz cross terms are linear in tau at fixed sigma_yz and sigma_yz^2 terms are constant.
  syz_correction() = the 5th component (0 for a v0.1 entry).
"""
import numpy as np

GPA_A2_TO_EV_A = 1e9 * 1e-20 / 1.602176634e-19 * 1e-10
SYZ_STEP_MAX = 0.25                                   # GPa, Ansh's Q1


def surface_force(sigma_gpa, Lx, Lz, n_slab):
    return sigma_gpa * Lx * Lz * GPA_A2_TO_EV_A / n_slab


def syz_ladder(target, step=SYZ_STEP_MAX):
    """sigma_yz values for the pre-ramp at tau = 0, from 0 (excluded) to target (included), steps <= step. [] for target 0."""
    if abs(target) < 1e-12:
        return []
    n = int(np.ceil(abs(target) / step - 1e-9))
    return [round(target * k / n, 6) for k in range(1, n + 1)]


def model_correction(entry, tau):
    """Correction vector at applied tau: 4 components (v0.1 entry) or 5 (v0.2 entry), see the module note."""
    c = None
    for p in range(5):
        if f"q{p}" in entry:
            t = np.asarray(entry[f"q{p}"], float) * tau ** p
            c = t if c is None else c + t
    return c


def syz_correction(c):
    return float(c[4]) if c is not None and len(c) > 4 else 0.0


def step_cmd(np_ranks, script, log, pot, datain, dataout, peout, fx, y0, Lx, Lz, xz, ys, zs, maxit, groups, halt_dist,
             etol, ftol, dmax, maxeval, fz=None):
    cmd = ["mpirun", "-np", str(np_ranks), "lmp", "-in", script, "-screen", "none", "-log", log,
           "-var", "pot", pot, "-var", "datain", datain, "-var", "dataout", dataout, "-var", "peout", peout,
           "-var", "fx", f"{fx:.10f}", "-var", "y0", f"{y0}", "-var", "Lx", f"{Lx:.10f}", "-var", "Lz", f"{Lz:.10f}",
           "-var", "xz", f"{xz:.10f}", "-var", "ys", f"{ys:.6f}", "-var", "zs", f"{zs:.6f}", "-var", "maxit", str(maxit),
           "-var", "groups", groups, "-var", "halt_dist", f"{halt_dist:.4f}",
           "-var", "etol", etol, "-var", "ftol", ftol, "-var", "dmax", dmax, "-var", "maxeval", str(maxeval)]
    if fz is not None and fz != 0.0:
        cmd += ["-var", "fz", f"{fz:.10f}", "-var", "fzb", f"{-fz:.10f}"]
    if fx < 0:
        cmd += ["-var", "fxb", f"{-fx:.10f}"]
    return cmd
