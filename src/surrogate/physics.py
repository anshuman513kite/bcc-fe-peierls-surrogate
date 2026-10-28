"""Physics-informed target (Session 12): learn the deviation from Schmid's law instead of tau_P itself.

r(chi) = tau_P(chi) * cos(chi) / tau0, i.e. the resolved shear on the (-110) plane at the Peierls stress, in units of a
constant tau0. Schmid's law (only the (-110) resolved shear matters) is r = 1 for every chi; the twinning/anti-twinning
asymmetry is r != 1. The wrapped model (GPBaseline or MLPEnsemble) is fitted on r; predictions are converted back:
tau_P = r * tau0 / cos(chi), spread s_tau = s_r * tau0 / cos(chi).
tau0 is only a scale: both base models standardise their targets (GP normalize_y with the noise level scaled alike, MLP
mean/std), so the tau_P predictions do not depend on it (checked in scripts/surrogate_loo.py), and leaving out chi = 0
in cross-validation does not leak the left-out value through tau0.
X[:, 0] must be chi in degrees.
"""
import numpy as np


class SchmidRatio:
    def __init__(self, base, tau0):
        self.base, self.tau0 = base, tau0

    def _c(self, X):
        return np.cos(np.radians(np.asarray(X, float).reshape(len(X), -1)[:, 0]))

    def fit(self, X, y):
        X = np.asarray(X, float).reshape(len(y), -1)
        self.base.fit(X, np.asarray(y, float) * self._c(X) / self.tau0)
        return self

    def predict(self, X, **kw):
        X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
        m, s = self.base.predict(X, **kw)
        f = self.tau0 / self._c(X)
        return m * f, s * f
