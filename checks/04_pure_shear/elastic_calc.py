"""Calculations (no simulation) in the screw-cell frame x = [111], y = [-110], z = [-1-12].

Uses the rotated stiffness C' from checks/03_tilt_xz/rotate_C.py (our measured C11, C12, C44). Voigt 1..6 = xx yy zz yz xz xy,
engineering shear strains.
(1) the xy column C'_i6; the slab response to surface-force sigma_xy (free y faces: sigma_yy = sigma_yz = 0, sigma_xy = applied;
    box fixed in x, z: eps_xx = eps_zz = gamma_xz = 0).
(2) option 2: strains for a PURE sigma_xz (all other stress components 0), from the compliance S = C'^-1;
    the same stress state in the slab needs the same box strains (eps_xx, eps_zz, gamma_xz); the other strains appear by themselves.
"""
import numpy as np
CV = np.load("../03_tilt_xz/Crot.npy")
S = np.linalg.inv(CV)
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
print("(1) xy column of C' (GPa): " + "  ".join(f"C'_{i + 1}6({n}) = {CV[i, 5]:+.4f}" for i, n in enumerate(names)))

free, fixed = [1, 3, 5], [0, 2, 4]          # slab: y faces free (tractions yy, yz, xy set), box fixes xx, zz, xz
tau = 1.0                                   # GPa
e = np.zeros(6)
# unknown strains e[free]; conditions: sigma_yy = 0, sigma_yz = 0, sigma_xy = tau (rows 1, 3, 5)
e[free] = np.linalg.solve(CV[np.ix_(free, free)], np.array([0, 0, tau]) - CV[np.ix_(free, fixed)] @ e[fixed])
s = CV @ e
print("    slab under surface-force sigma_xy = 1 GPa: strains " + " ".join(f"{n}={v:+.6f}" for n, v in zip(names, e)))
print("    stresses (GPa): " + " ".join(f"{n}={v:+.6f}" for n, v in zip(names, s)))
print(f"    effective modulus sigma_xy/gamma_xy = {tau / e[5]:.3f} GPa (C'_66 = {CV[5, 5]:.3f}, minus C'_46^2/C'_44 = {CV[3, 5]**2 / CV[3, 3]:.3f})")

print("(2) compliance S = C'^-1, column for sigma_xz (strain per GPa of pure sigma_xz):")
col = S[:, 4]
for n, v in zip(names, col):
    print(f"    {'eps' if n[0] == n[1] else 'gamma'}_{n} = {v:+.7f} per GPa")
for t in (0.357, 0.714):
    print(f"    sigma_xz = {t} GPa: " + " ".join(f"{n}={v * t:+.7f}" for n, v in zip(names, col)))
# slab: box strains eps_xx, eps_zz, gamma_xz from the same pure-stress state; check by the slab solve
for t in (0.357, 0.714):
    e = np.zeros(6); e[fixed] = col[fixed] * t
    e[free] = np.linalg.solve(CV[np.ix_(free, free)], -CV[np.ix_(free, fixed)] @ e[fixed])
    print(f"    slab, box strains xx={e[0]:+.7f} zz={e[2]:+.7f} xz={e[4]:+.7f} -> stresses "
          + " ".join(f"{n}={v:+.6f}" for n, v in zip(names, CV @ e)) + f" ; free strains yy={e[1]:+.7f} yz={e[3]:+.7f} xy={e[5]:+.7f}")
np.save("S_col_xz.npy", col)
