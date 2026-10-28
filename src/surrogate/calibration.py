"""Calibration check: fraction of true values inside mean +- 1 and +- 2 predicted spreads (ideal Gaussian: 0.683 and 0.954)."""
import numpy as np


def coverage(y_true, mean, std):
    z = np.abs(np.asarray(y_true) - np.asarray(mean)) / np.maximum(np.asarray(std), 1e-12)
    return dict(within_1sigma=float(np.mean(z <= 1)), within_2sigma=float(np.mean(z <= 2)), ideal_1=0.683, ideal_2=0.954)
