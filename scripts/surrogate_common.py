"""Shared pieces for the surrogate on the REAL chi-sweep data (Session 12): dataset, the four model variants, helpers.

Dataset v1 (data/surrogate_train_v1.csv): the 10 small-cell sweep angles (data/peierls_chi_sweep.csv, bracket midpoints) plus
the chi = 0 baseline (Gate B', small cell, 1.285-1.290 GPa); 11 points, all in the 146.87 A cell. The 2x-cell rows (chi = +-30)
are not used: a different cell. Noise sigma = 0.010 / sqrt(12) = 0.0029 GPa for every point (bracket width / sqrt 12; the
chi = 0 bracket is 5 MPa wide, so 0.0029 is conservative there).
Models (fixed settings from Sessions 10-11, nothing tuned on this data):
  gp_tau    GPBaseline (constant x RBF, white noise fixed to sigma) on tau_P
  mlp_tau   MLPEnsemble (5 members, tanh 32x32, bootstrap + randomized priors, 2000 epochs), noise sigma added in quadrature
  gp_ratio  / mlp_ratio: the same models on r = tau_P cos(chi) / tau0 (SchmidRatio), converted back to tau_P
  gp_tau_d4 / gp_ratio_d4 (Ansh's D4, Session 13; the model for the rest of live active learning): GP with the white-noise
            level LEARNED within [0.0029, 0.1] GPa (in r units: divided by tau0). MLP unchanged.
"""
import csv, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from surrogate import MLPEnsemble, GPBaseline, SchmidRatio, BRACKET_UNCERTAINTY_GPA
SIGMA = BRACKET_UNCERTAINTY_GPA          # 0.0029 GPa
TAU0 = 1.315                             # chi = 0 baseline midpoint (D6: B2, sweep protocol); only a scale for the ratio models
TRAIN_V1 = os.path.join(ROOT, "data", "surrogate_train_v1.csv")
LIVE_RESULTS = os.path.join(ROOT, "data", "al_live_results.csv")
MODELS = ("gp_tau", "mlp_tau", "gp_ratio", "mlp_ratio")
NOISE_BOUNDS = (SIGMA, 0.1)             # GPa, D4


class _NoisyMLP(MLPEnsemble):
    """MLPEnsemble whose predict() includes the measurement noise (spread of a new measurement), like the GP's."""
    def predict(self, X, include_noise=True):
        return super().predict(X, include_noise=include_noise)


def make_model(name, seed=0, tau0=TAU0):
    if name == "gp_tau":
        return GPBaseline(seed=seed, noise_std=SIGMA)
    if name == "mlp_tau":
        return _NoisyMLP(seed=seed, noise_std=SIGMA)
    if name == "gp_ratio":
        return SchmidRatio(GPBaseline(seed=seed, noise_std=SIGMA / tau0), tau0)
    if name == "gp_tau_d4":
        return GPBaseline(seed=seed, noise_bounds=NOISE_BOUNDS)
    if name == "gp_ratio_d4":
        return SchmidRatio(GPBaseline(seed=seed, noise_bounds=tuple(b / tau0 for b in NOISE_BOUNDS)), tau0)
    if name == "mlp_ratio":
        return SchmidRatio(_NoisyMLP(seed=seed, noise_std=SIGMA / tau0), tau0)
    raise ValueError(name)


def learned_noise_gpa(model, tau0=TAU0):
    """Fitted GP noise in GPa (for the ratio model: r units times tau0, i.e. the noise in tau_P at chi = 0)."""
    return model.base.learned_noise() * tau0 if isinstance(model, SchmidRatio) else model.learned_noise()


def load_train(include_live=True):
    """chi (deg), tau_P (GPa) of v1, plus the live active-learning results if present."""
    rows = list(csv.DictReader(open(TRAIN_V1)))
    chi = [float(r["chi_deg"]) for r in rows]; tau = [float(r["tau_mid"]) for r in rows]
    if include_live and os.path.exists(LIVE_RESULTS):
        for r in csv.DictReader(open(LIVE_RESULTS)):
            if r.get("rule") == "D5":                  # only rows recorded under the current rule (D5, Session 14) are used
                chi.append(float(r["chi_deg"])); tau.append(float(r["tau_mid"]))
    o = np.argsort(chi)
    return np.array(chi)[o], np.array(tau)[o]
