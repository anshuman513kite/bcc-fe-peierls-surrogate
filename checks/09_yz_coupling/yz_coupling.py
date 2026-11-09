"""Calculation only (no simulation), 2 Nov 2026: surface-force sigma_yz = 1 GPa on the free y faces of the slab, box strains
(xx, zz, xz) unchanged. Rotated stiffness from checks/02_elastic (e = 0.001), frame x = [111], y = [-110], z = [-1-12].
Voigt order xx yy zz yz xz xy, engineering shear strains."""
import numpy as np
C11, C12, C44 = 243.3297, 145.0091, 115.9907
C = np.zeros((3,3,3,3))
for i in range(3):
 for j in range(3):
  for k in range(3):
   for l in range(3):
    C[i,j,k,l] = C12*(i==j)*(k==l) + C44*((i==k)*(j==l)+(i==l)*(j==k)) + (C11-C12-2*C44)*(i==j==k==l)
R = np.array([[1,1,1],[-1,1,0],[-1,-1,2]], float); R /= np.linalg.norm(R,axis=1)[:,None]
Cr = np.einsum("ai,bj,ck,dl,ijkl->abcd", R,R,R,R,C)
voigt=[(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]; names=["xx","yy","zz","yz","xz","xy"]
CV = np.array([[Cr[a][b][c][d] for (c,d) in voigt] for (a,b) in voigt])
print("column 4 (gamma_yz -> sigma_i):", " ".join(f"{n}={v:+.3f}" for n,v in zip(names, CV[:,3])))
# slab: box strains (xx, zz, xz) fixed at 0; y-face tractions: syy=0, syz=1 GPa, sxy=0
free, fixed = [1,3,5], [0,2,4]
target = np.array([0.0, 1.0, 0.0])         # syy, syz, sxy
e = np.zeros(6); e[free] = np.linalg.solve(CV[np.ix_(free,free)], target)
s = CV @ e
print("slab, 1 GPa sigma_yz by surface forces, box strains unchanged:")
print("  strains:", " ".join(f"{n}={v:+.6f}" for n,v in zip(names,e)))
print("  stresses:", " ".join(f"{n}={v:+.5f}" for n,v in zip(names,s)))
