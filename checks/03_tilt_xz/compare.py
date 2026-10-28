"""Compare the LAMMPS stress change sigma(gamma) - sigma(0) with the rotated-tensor prediction C'_i5 * gamma."""
import re, numpy as np
CV = np.load("Crot.npy")
names = ["xx", "yy", "zz", "yz", "xz", "xy"]
def stress(g):
    m = re.search(r"^STRESS_GPa (.*)$", open(f"log.tilt_{g}").read(), re.M).group(1)
    d = dict(kv.split("=") for kv in m.split())
    return np.array([float(d["s" + n]) for n in names])
s0 = stress("0.0")
print(f"gamma = 0: " + " ".join(f"s{n}={v:+.6f}" for n, v in zip(names, s0)) + " GPa (residual)")
for g in ["0.005", "0.010", "-0.005"]:
    ds = stress(g) - s0
    pred = CV[:, 4] * float(g)
    print(f"gamma = {g}:")
    for n, v, p in zip(names, ds, pred):
        print(f"   d_sigma_{n} = {v:+.6f} GPa   predicted C'_i5*gamma = {p:+.6f}   ratio to d_sigma_xz = {v / ds[4] * 100:+7.2f} %")
    print(f"   sigma_xz vs G*gamma (G = {CV[4, 4]:.3f}): {(ds[4] / pred[4] - 1) * 100:+.2f} %")
