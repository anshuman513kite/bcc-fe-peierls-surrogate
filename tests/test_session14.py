"""Session 14 tests: Ansh's D5 "moved" rule (label-free) and the stability test's same-state check; TAU0 scale invariance."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
from moved_rule import classify, is_moved, same_state


def st(dy, dz, orient, top, col):
    return dict(dy=dy, dz=dz, orient=orient, topcol=top, col=col)


def test_split_is_moved():
    ev = classify(1.346, 0.0, "up", 1.80)                 # chi = +28 at 2.120 GPa: split core on a vertex column
    assert ev == "split" and is_moved(ev)
    ev = classify(2.019, 1.1657, "down", 1.545)           # chi = -28 first move: same kind of state, other triangle label
    assert ev == "split" and is_moved(ev)


def test_relabelling_is_not_a_new_event():
    a = st(1.346, 0.0, "up", 1.803, (0.67, 1.17))         # +28, 2.160 GPa
    b = st(2.019, 1.1657, "down", 1.804, (0.67, 1.17))    # +28, 2.170 GPa: atoms moved <= 0.013 A, label flipped
    assert same_state(a, b)
    assert classify(a["dy"], a["dz"], a["orient"], a["topcol"]) == classify(b["dy"], b["dz"], b["orient"], b["topcol"]) == "split"


def test_glide_jump_to_compact_site_is_moved():
    ev = classify(0.0, 2.3313, "down", 1.21)
    assert ev == "glide_jump" and is_moved(ev)
    assert classify(-6.057, 3.497, "down", 1.159) == "glide_jump"      # chi = +30: 3 jumps on (-1 0 1)


def test_compact_at_start_is_not_moved():
    assert classify(0.0, 0.0, "down", 1.225) == "none"
    assert not is_moved("none") and not is_moved("anomaly")
    assert same_state(st(0, 0, "down", 1.20, (0.67, 1.17)), st(0, 0.004, "down", 1.21, (0.67, 1.17)))
    assert not same_state(st(0, 0, "down", 1.20, (0.67, 1.17)), st(0, 2.3313, "down", 1.21, (0.67, 3.50)))
    assert not same_state(st(0, 0, "down", 1.22, (0.67, 1.17)), st(1.346, 0, "up", 1.80, (0.67, 1.17)))


def test_ratio_model_scale_invariant_tau0():
    """D6 sets TAU0 = 1.315: the ratio models' tau_P predictions must not depend on it."""
    from surrogate_common import make_model
    chi = np.array([-30, -18, -6, 0, 6, 18, 30], float); tau = np.array([1.165, 1.215, 1.255, 1.315, 1.395, 1.645, 2.045])
    for name in ("gp_ratio", "gp_ratio_d4"):
        a = make_model(name, tau0=1.315).fit(chi[:, None], tau).predict(np.array([[12.0]]))
        b = make_model(name, tau0=1.0).fit(chi[:, None], tau).predict(np.array([[12.0]]))
        assert abs(a[0][0] - b[0][0]) < 1e-6 and abs(a[1][0] - b[1][0]) < 1e-6
