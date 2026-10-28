"""A made-up 1D test curve with noise. SYNTHETIC: never mix with LAMMPS data; it only resembles the shape of tau_P(chi)."""
import numpy as np


def synthetic_curve(x, noise=0.0, seed=None):
    """SYNTHETIC test function on x in degrees (about -30..30): smooth, asymmetric, not a physical model.
    Returns y (and adds Gaussian noise of std `noise` if noise > 0)."""
    x = np.asarray(x, float)
    y = 1.3 / np.cos(np.radians(x)) + 0.15 * np.sin(np.radians(3 * x)) + 0.05 * np.tanh(x / 8)
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.shape)
    return y
