"""One Newton correction for option 2: measured interior stress (minus the zero-strain reference) vs the pure target;
block: delta_eps = S (target - measured), all six strains; slab: only the box strains xx, zz, xz can be set (y faces free),
delta = K^-1 (target - measured) on (xx, zz, xz), with K the slab stiffness for box strains. Prints the corrected strain sets."""
import re, numpy as np
CV = np.load("../03_tilt_xz/Crot.npy"); S = np.linalg.inv(CV)
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
fr, fx = [1, 3, 5], [0, 2, 4]
K = CV[np.ix_(fx, fx)] - CV[np.ix_(fx, fr)] @ np.linalg.solve(CV[np.ix_(fr, fr)], CV[np.ix_(fr, fx)])
def interior(tag):
    d = dict(kv.split("=") for kv in re.search(r"^INTERIOR_GPa \S+ (.*)$", open(f"log.{tag}").read(), re.M).group(1).split())
    return np.array([float(d["s" + n]) for n in names])
base = {"block": interior("block_ref"), "slab": interior("slab_ref")}
cases = {"block_pure0.357": [0, -0.00098621, 0.00098621, 0, 0.00586727, 0], "block_pure0.714": [0, -0.00197243, 0.00197243, 0, 0.01173452, 0],
         "slab_pure0.357": [0, 0, 0.00098621, 0, 0.00586727, 0], "slab_pure0.714": [0, 0, 0.00197243, 0, 0.01173452, 0]}
# Voigt order of strains here: exx eyy ezz gyz gxz gxy
for tag, e in cases.items():
    geom = tag.split("_")[0]; t = float(tag.split("pure")[1])
    target = np.zeros(6); target[4] = t
    res = interior(tag) - base[geom] - target
    e = np.array(e, float)
    if geom == "block":
        e = e - S @ res
    else:
        e[fx] = e[fx] - np.linalg.solve(K, res[fx])
    print(f"{tag}c " + " ".join(f"{v:.8f}" for v in e))
