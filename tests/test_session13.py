"""Session 13 tests: Ansh's D1 ("moved" rule), D2 (Gate A sigma_xz criterion), D4 (learned GP noise floor), and the
Vitek-type formula used for the Chaussidon benchmark."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from moved_rule import classify_d1 as classify, JUMP     # D1 (Session 13), kept as history; D5 tests in test_session14.py
from gate_a import gate_a_check
from surrogate import GPBaseline
from fit_vitek import vitek_tau


def test_d1_core_change_is_not_moved():
    assert classify(1.346, 0.0, "up") == "core_change"           # chi = +28 event: neighbouring triangle, < one jump
    assert classify(0.0, 2.3313, "down") == "moved"              # one jump along the glide direction
    assert classify(2.019, 1.1657, "down") == "moved"            # one jump onto the +60 deg plane (chi = -24 / -28 / -30)
    assert classify(0.0, 0.005, "down") == "none"
    assert abs(JUMP - 2.3313) < 1e-4


def _stress(chi, tau, dxz):
    s = np.zeros(6); s[5] = tau * np.cos(np.radians(chi)); s[4] = tau * np.sin(np.radians(chi)) + dxz
    return s


def test_d2_angle_criterion():
    *_, rxz, ang, _, ok = gate_a_check(_stress(3, 0.5, 0.0003), 3, 0.5)
    assert rxz > 0.01 and ok                                     # 1.1 % of target, but a 0.034 deg tilt: passes under D2
    *_, ok_bad = gate_a_check(_stress(3, 0.5, 0.01), 3, 0.5)
    assert not ok_bad                                            # 0.02 rad = 1.15 deg: fails
    *_, ok_old = gate_a_check(_stress(30, 1.5, 0.005), 30, 1.5)
    assert ok_old                                                # within 1 % of a large target: still passes (OR)


def test_d4_learned_noise_floor():
    rng = np.random.default_rng(0)
    x = np.linspace(-30, 30, 11)
    for y in (1.3 + 0.01 * x, 1.3 + 0.01 * x + 0.05 * rng.standard_normal(11)):   # smooth (noise -> floor) and noisy data
        gp = GPBaseline(noise_bounds=(0.0029, 0.1)).fit(x[:, None], y)
        assert gp.learned_noise() >= 0.0029 * (1 - 1e-6)
        assert gp.learned_noise() <= 0.1 * (1 + 1e-6)


def test_vitek_formula_chaussidon_chi0():
    assert abs(vitek_tau(0.0, 1604.0, 0.61) - 1229.0) <= 1.0     # MPa
