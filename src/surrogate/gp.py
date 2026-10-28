"""Gaussian-process baseline (scikit-learn): RBF kernel with a learned length scale plus a white-noise term.
noise_std (target units, e.g. the Peierls-bracket uncertainty in GPa): if given, the white-noise level is FIXED to it
instead of learned.
noise_bounds = (lo, hi) (target units; Ansh's D4, Session 13): the white-noise level is LEARNED but kept within [lo, hi],
e.g. (0.0029, 0.1) GPa: never below the bracket noise. Overrides noise_std. learned_noise() returns the fitted level.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class GPBaseline:
    def __init__(self, seed=0, noise_std=None, noise_bounds=None):
        self.seed, self.noise_std, self.noise_bounds = seed, noise_std, noise_bounds

    def fit(self, X, y):
        X = np.asarray(X, float).reshape(len(y), -1)
        self.y_std = np.std(y) + 1e-12             # normalize_y=True works in units of the target std
        if self.noise_bounds:
            lo, hi = (b / self.y_std for b in self.noise_bounds)
            w = WhiteKernel(lo ** 2, (lo ** 2, hi ** 2))
        elif self.noise_std:
            w = WhiteKernel((self.noise_std / self.y_std) ** 2, "fixed")
        else:
            w = WhiteKernel(1e-3, (1e-8, 1e0))
        k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + w
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=3, random_state=self.seed).fit(X, y)
        return self

    def learned_noise(self):
        """Fitted white-noise standard deviation in target units."""
        return float(np.sqrt(self.gp.kernel_.k2.noise_level) * self.y_std)

    def predict(self, X):
        X = np.asarray(X, float).reshape(-1, self.gp.X_train_.shape[1])
        m, s = self.gp.predict(X, return_std=True)
        return m, s
