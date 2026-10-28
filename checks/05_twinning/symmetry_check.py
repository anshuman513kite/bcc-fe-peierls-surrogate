"""Independent check of twin_sense.py: carry the textbook twinning system (112)[-1-11] (twinning shear of the +n side along
[-1-11]) onto our planes with every cubic symmetry operation (signed permutation matrix) that maps (112) onto the plane."""
import itertools
import numpy as np
n0, d0 = np.array([1, 1, 2]), np.array([-1, -1, 1])
for name, n in [("chi = +30, (-2 1 1)", (-2, 1, 1)), ("chi = -30, (-1 2 -1)", (-1, 2, -1))]:
    found = set()
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), int)
            for i in range(3):
                R[i, p[i]] = sg[i]
            if (R @ n0 == np.array(n)).all() and abs(R @ d0 @ np.array([1, 1, 1])) == 3:   # image direction is +-[111]
                found.add(tuple(R @ d0))
    print(f"{name}: twinning direction of the +n side (images of [-1-11]) = {sorted(found)}")
