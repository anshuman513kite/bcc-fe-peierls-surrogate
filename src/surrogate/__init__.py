"""Project A surrogate-model code (Track 2): ensemble of small MLPs, GP baseline, active learning, calibration check.

Inputs are arrays X of shape (n, d): d = 1 now (chi), d = 2 later (chi, sigma_yz). Targets y of shape (n,).
Tested on clearly labelled SYNTHETIC curves (synthetic.py); from Session 12 also used on the real chi-sweep data
(scripts/surrogate_*.py), never mixed with synthetic data.
"""
from .ensemble import MLPEnsemble
from .gp import GPBaseline
from .active import active_learning, compare_strategies
from .calibration import coverage
from .synthetic import synthetic_curve
from .physics import SchmidRatio

# Measured Peierls-bracket uncertainty (GPa), the target noise level option (noise_std of MLPEnsemble / GPBaseline).
# Gate B' (Session 11): brackets identical on 7 and 8 MPI ranks and reproduced by 30/30 stability tests, so the uncertainty
# is the bracket width w itself; uniform over the bracket: std = w / sqrt(12). Sweep refinement w = 0.010 GPa.
BRACKET_WIDTH_GPA = 0.010
BRACKET_UNCERTAINTY_GPA = BRACKET_WIDTH_GPA / 12 ** 0.5     # = 0.0029 GPa
