"""Rotated stiffness of bcc Fe in the screw-cell frame x = [111], y = [-110], z = [-1-12].

C11, C12, C44 from checks/02_elastic (e = 0.001). Voigt order 1..6 = xx, yy, zz, yz, xz, xy; engineering shear strains.
Prints the column of C' that a pure shear gamma_xz (Voigt 5) produces: sigma_i = C'_i5 * gamma_xz.
"""
import numpy as np
C11, C12, C44 = 243.3297, 145.0091, 115.9907          # GPa, checks/02_elastic/log.elastic_e0.001
C = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C[i, j, k, l] = (C12 * (i == j) * (k == l) + C44 * ((i == k) * (j == l) + (i == l) * (j == k))
                                 + (C11 - C12 - 2 * C44) * (i == j == k == l))
R = np.array([[1, 1, 1], [-1, 1, 0], [-1, -1, 2]], float)
R /= np.linalg.norm(R, axis=1)[:, None]                 # rows = new axes in cubic coordinates
Cr = np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, C)
voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
CV = np.array([[Cr[a][b][c][d] for (c, d) in voigt] for (a, b) in voigt])
print("rotated stiffness C' (GPa), Voigt xx yy zz yz xz xy:")
for n, row in zip(names, CV):
    print(f"  {n}: " + " ".join(f"{v:9.3f}" for v in row))
print(f"C'_55 (sigma_xz per gamma_xz) = {CV[4, 4]:.4f} GPa ; (C11-C12+C44)/3 = {(C11 - C12 + C44) / 3:.4f} GPa")
for i, n in enumerate(names):
    if i != 4:
        print(f"  coupling gamma_xz -> sigma_{n}: C'_{i + 1}5 = {CV[i, 4]:+.4f} GPa  ({CV[i, 4] / CV[4, 4] * 100:+.2f} % of C'_55)")
np.save("Crot.npy", CV)
