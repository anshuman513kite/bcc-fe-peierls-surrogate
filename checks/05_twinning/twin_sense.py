"""Twinning vs anti-twinning sense on the {112} planes of the [111] zone, in our frame (geometry only, no simulation).

Criterion (stacking argument): {112} planes of bcc stack in a 6-layer sequence; going from one layer to the next (towards +n),
the in-plane offset along b = a/2[111] is c*b with c = 1/3 or 2/3 (mod 1). The twin is the mirror image across the plane,
whose per-layer offset is -c. Turning the perfect stacking into the twin needs an extra per-layer shift delta = -2c (mod 1);
the twinning sense is the SHORT one, |delta| = 1/3 (i.e. a/6<111>, the twinning shear); shearing the crystal on the +n side
the other way (needs 2/3, a/3<111>) is the anti-twinning sense.
Loading: MRSS plane normal n = cos(chi) y + sin(chi) z; traction on it along x = sigma_xy n_y + sigma_xz n_z = tau > 0,
so the crystal on the +n side is sheared along +x = +[111] relative to the -n side.
"""
import numpy as np
ex = np.array([1, 1, 1]) / np.sqrt(3); ey = np.array([-1, 1, 0]) / np.sqrt(2); ez = np.array([-1, -1, 2]) / np.sqrt(6)


def twin_dir(n, d=(1, 1, 1)):
    """Plane normal n (integer, h+k+l even) and a <111> direction d lying in the plane.
    Return (c, sense, t): sense = +1 if twinning moves the +n side along +d, -1 if along -d."""
    n = np.array(n); B = np.array(d) / 2                    # b = a/2 d, in units of a
    assert n @ np.array(d) == 0, "d must lie in the plane"
    # lattice vector t joining one layer to the next: n.t = 1 (corner sites are integer vectors; try small ones)
    t = next(np.array(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)] if n @ np.array(v) == 1)
    c = (t @ B) / (B @ B) % 1.0                                # offset along b per layer, in units of b
    delta = (-2 * c) % 1.0                                     # extra per-layer shift (mod 1) that makes the mirror stacking
    sense = +1 if abs(delta - 1 / 3) < 1e-9 else -1            # +1/3 b -> twinning moves the +n side along +b
    assert abs(min(delta, 1 - delta) - 1 / 3) < 1e-9
    return c, sense, t


# textbook check: bcc twinning system (112)[-1-11], i.e. on (112) the +n side moves along [-1-11] = -[11-1]
c, s, t = twin_dir((1, 1, 2), (1, 1, -1))
print(f"check (112), d = [11-1]: c = {c:.4f}; twinning moves the +n side along {'+' if s > 0 else '-'}[11-1]"
      f" -> {'[-1-11], matches the textbook (112)[-1-11]' if s < 0 else '[11-1], does NOT match the textbook'}")
for chi in (+30, -30):
    nv = np.cos(np.radians(chi)) * ey + np.sin(np.radians(chi)) * ez
    n = np.round(nv / np.abs(nv).max() * 2).astype(int)      # Miller indices of the plane, oriented as the MRSS normal
    assert np.allclose(nv, n / np.linalg.norm(n))
    c, s, t = twin_dir(n)
    print(f"chi = {chi:+d} deg: plane ({n[0]:d} {n[1]:d} {n[2]:d}), normal in (y, z) = ({nv @ ey:+.3f}, {nv @ ez:+.3f}); "
          f"stacking offset c = {c:.4f} b per layer; twinning moves the +n side along {'+' if s > 0 else '-'}[111]; "
          f"our loading moves it along +[111] -> {'TWINNING' if s > 0 else 'ANTI-TWINNING'} sense")
