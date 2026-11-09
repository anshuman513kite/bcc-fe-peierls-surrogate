"""SYNTHETIC 2D test surfaces for the v0.2 pool-size study (D-v02-2), written 2 Nov 2026 before the study ran.
Never mix with LAMMPS data. Inputs: chi in degrees (-30..30), sigma in units of s (the sigma_yz range, nominal s = 1).
Formulas (also in V02_PLAN.md section 8):
  glide surface at sigma = 0: Vitek form tg0(chi) = S0 / (cos chi + A cos(60 deg + chi)), S0 = 1.7754 GPa, A = 0.70
      -> tg0(-30) = 1.206, tg0(0) = 1.315, tg0(+30) = 2.050 GPa (twinning / anti-twinning asymmetry of v0.1)
  split surface at sigma = 0: ts0 = tg0 + D(chi), D = 0.08 - 0.126 exp(-((chi + 30)/10)^2) - 0.15 exp(-((chi - 27)/3)^2) GPa
      -> split below glide for chi <~ -24 (ts0(-30) = 1.160) and for 25 <~ chi <~ 29 (as +28 in v0.1), glide elsewhere
  sigma effect (common factor and a split-only factor, both asymmetric in the sign of sigma):
      F(chi, sigma) = 1 + alpha(chi) (sigma + 0.5 sigma^2) / 1.5,   alpha(chi) = alpha0 (0.75 + chi / 120)
      tg = tg0 F,   ts = ts0 F (1 - (alpha0 / 6) sigma)
      alpha0 = 0.05 (weak), 0.15 (medium), 0.30 (strong)
  tau_P = min(tg, ts); event = 'glide' if tg <= ts else 'split'
  quantised to the 10 MPa bracket: floor(tau_P / 0.01) * 0.01 + 0.005 (the bracket midpoint, as the real ramps report)
"""
import numpy as np

S0, A = 1.7754, 0.70
ALPHA0 = {"weak": 0.05, "medium": 0.15, "strong": 0.30}
BRACKET = 0.01


def tg0(chi):
    c = np.radians(np.asarray(chi, float))
    return S0 / (np.cos(c) + A * np.cos(np.radians(60.0) + c))


def split_offset(chi):
    chi = np.asarray(chi, float)
    return 0.08 - 0.126 * np.exp(-((chi + 30) / 10) ** 2) - 0.15 * np.exp(-((chi - 27) / 3) ** 2)


def surfaces(chi, sig, version):
    a0 = ALPHA0[version]
    chi, sig = np.asarray(chi, float), np.asarray(sig, float)
    F = 1 + a0 * (0.75 + chi / 120) * (sig + 0.5 * sig ** 2) / 1.5
    g = tg0(chi) * F
    s = (tg0(chi) + split_offset(chi)) * F * (1 - a0 / 6 * sig)
    return g, s


def truth(chi, sig, version, quantise=True):
    """(tau_P, event) with event 'glide' / 'split'; tau_P quantised to the 10 MPa bracket midpoint by default."""
    g, s = surfaces(chi, sig, version)
    tau = np.minimum(g, s)
    if quantise:
        tau = np.floor(tau / BRACKET + 1e-9) * BRACKET + BRACKET / 2
    return tau, np.where(g <= s, "glide", "split")
