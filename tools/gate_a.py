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
