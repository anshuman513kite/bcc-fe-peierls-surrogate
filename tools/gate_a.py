"""Gate A pass test for one measured loading (perfect slab), shared by the calibration scripts.

Criteria (Gate A, Session 10; sigma_xz criterion extended by Ansh's D2, 28 Oct, Session 13):
  sigma_xy within 1 % of tau cos(chi);
  sigma_xz within 1 % of tau sin(chi) OR loading-angle error |d sigma_xz| / tau <= 0.1 deg (0.001745 rad)   [D2]
  every other component (xx, yy, zz, yz) below 2 % of tau.
D2 was introduced AFTER the chi = +3 calibration failed (tau = 0.5: |d sigma_xz| = 0.0003 GPa on a 0.026 GPa target, 1.23 %,
which tilts the loading by only 0.034 deg). It is an OR, so no point that passed before can fail now.
s, the measured stress, is ordered (xx, yy, zz, yz, xz, xy) in GPa.
"""
import numpy as np

ANGLE_TOL_RAD = np.radians(0.1)       # 0.001745 rad


def gate_a_check(s, chi_deg, tau):
    t = np.zeros(6); t[5] = tau * np.cos(np.radians(chi_deg)); t[4] = tau * np.sin(np.radians(chi_deg))
    err = np.asarray(s, float) - t
    rxy = abs(err[5]) / abs(t[5])
    rxz = abs(err[4]) / abs(t[4]) if abs(t[4]) > 1e-12 else abs(err[4]) / tau
    ang = abs(err[4]) / tau                                    # loading-angle error, rad
    oth = max(abs(err[[0, 1, 2, 3]])) / tau
    ok = rxy < 0.01 and (rxz < 0.01 or ang <= ANGLE_TOL_RAD) and oth < 0.02
    return err, rxy, rxz, ang, oth, ok


FLOOR_REF_GPA = 1.0                   # V02_PLAN 8.8: off-axis tolerances never below 0.001745 * 1.0 GPa


def gate_a3_check(s, chi_deg, tau, syz):
    """v0.2 Gate A with three loaded components (Ansh, V02_PLAN section 8 Q3, 2 Nov 2026). s ordered (xx, yy, zz, yz, xz, xy), GPa.
    ref = max(tau, |syz|), the largest applied stress.
      sigma_xy: within 1 % of tau cos(chi)                         [tau > 0; at tau = 0 see below]
      sigma_xz: within 1 % of tau sin(chi) OR |d sigma_xz| / tau <= 0.001745 (D2)
      sigma_yz: within 1 % of syz OR |d sigma_yz| <= 0.001745 * ref     (Q3, option a; also covers syz = 0)
      xx, yy, zz: each < 2 % of tau; at tau = 0 points < 2 % of |syz|  (Q3)
    At tau = 0 the sigma_xy and sigma_xz targets are zero and the 1 %/D2 forms are undefined; they are then held to the same
    D2-style floor as sigma_yz, |d sigma| <= 0.001745 * |syz| (Claude's reading, to be confirmed by Ansh).
    V02_PLAN 8.8 (3 Nov 2026, Ansh): every off-axis absolute tolerance (sigma_xz angle branch, sigma_yz floor, sigma_xy and
    sigma_xz at tau = 0) is 0.001745 * max(tau, |syz|, 1.0 GPa): unchanged for tau >= 1.0 GPa (|syz| <= 1.0), held at
    0.0017 GPa below. The strain model's absolute error is ~0.001 GPa at every tau, so a tolerance proportional to tau
    failed by construction below ~0.6 GPa (smoke test S1, fc0cb2f). The 1 % branches and the 2 % rule are unchanged.
    Returns a dict with the relative / absolute errors and ok. tau = syz = 0 has nothing loaded and raises ValueError."""
    if tau <= 0 and abs(syz) < 1e-12:
        raise ValueError("nothing loaded (tau = 0 and sigma_yz = 0): not a Gate A point")
    ref = max(tau, abs(syz))
    t = np.zeros(6); t[5] = tau * np.cos(np.radians(chi_deg)); t[4] = tau * np.sin(np.radians(chi_deg)); t[3] = syz
    err = np.asarray(s, float) - t
    floor = ANGLE_TOL_RAD * max(ref, FLOOR_REF_GPA)       # 8.8
    if tau > 0:
        rxy = abs(err[5]) / abs(t[5])
        ok_xy = rxy < 0.01
        rxz = abs(err[4]) / abs(t[4]) if abs(t[4]) > 1e-12 else np.inf
        ang = abs(err[4]) / tau
        ok_xz = rxz < 0.01 or abs(err[4]) <= floor              # 8.8 (before: ang <= ANGLE_TOL_RAD, i.e. 0.001745 * tau)
    else:
        rxy = rxz = np.inf; ang = abs(err[4]) / ref
        ok_xy, ok_xz = abs(err[5]) <= floor, abs(err[4]) <= floor
    ryz = abs(err[3]) / abs(syz) if abs(syz) > 1e-12 else np.inf
    ok_yz = ryz < 0.01 or abs(err[3]) <= floor
    oth = max(abs(err[[0, 1, 2]])) / (tau if tau > 0 else abs(syz))
    ok_oth = oth < 0.02
    return dict(err=err, rxy=rxy, rxz=rxz, ang=ang, ryz=ryz, yz_abs=abs(err[3]), floor=floor, oth=oth,
                ok_xy=ok_xy, ok_xz=ok_xz, ok_yz=ok_yz, ok_oth=ok_oth, ok=bool(ok_xy and ok_xz and ok_yz and ok_oth))
