"""Calculation only (no simulation): stress from a pure xz tilt when the y faces are free (sigma_yy = sigma_yz = sigma_xy = 0),
as in the screw-cell slab. eps_xx = eps_zz = 0 (periodic x, z), gamma_xz = gamma; eps_yy, gamma_yz, gamma_xy free."""
import numpy as np
CV = np.load("Crot.npy"); names = ["xx", "yy", "zz", "yz", "xz", "xy"]
free, fixed = [1, 3, 5], [0, 2, 4]
e = np.zeros(6); e[4] = 1.0                                     # per unit gamma
e[free] = np.linalg.solve(CV[np.ix_(free, free)], -CV[np.ix_(free, fixed)] @ e[fixed])
s = CV @ e
print("free y faces, per unit gamma_xz: strains " + " ".join(f"{n}={v:+.4f}" for n, v in zip(names, e)))
print("  stresses (GPa per unit gamma): " + " ".join(f"{n}={v:+.3f}" for n, v in zip(names, s)))
print(f"  effective G = sigma_xz/gamma = {s[4]:.3f} GPa ; sigma_zz/sigma_xz = {s[2]/s[4]:+.3f} ; sigma_xx/sigma_xz = {s[0]/s[4]:+.3f}")
