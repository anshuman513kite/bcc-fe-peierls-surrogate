"""ML code runs on the SYNTHETIC curve (small, fast settings; checks that the code works, not model quality)."""
import numpy as np
import torch
from surrogate import MLPEnsemble, GPBaseline, active_learning, coverage, synthetic_curve

torch.set_num_threads(1)
X = np.linspace(-30, 30, 9)[:, None]
Y = synthetic_curve(X[:, 0], noise=0.01, seed=0)
G = np.linspace(-30, 30, 31)[:, None]


def test_ensemble_mean_and_spread():
    m, s = MLPEnsemble(n_members=3, epochs=200).fit(X, Y).predict(G)
    assert m.shape == s.shape == (31,) and np.all(np.isfinite(m)) and np.all(s >= 0) and s.max() > 0


def test_ensemble_two_inputs():
    X2 = np.column_stack([X[:, 0], np.linspace(-0.5, 0.5, len(X))])          # (chi, sigma_yz) layout for later
    m, s = MLPEnsemble(n_members=2, epochs=100).fit(X2, Y).predict(X2)
    assert m.shape == (len(X2),)


def test_gp_baseline():
    m, s = GPBaseline().fit(X, Y).predict(G)
    assert np.sqrt(np.mean((m - synthetic_curve(G[:, 0])) ** 2)) < 0.1 and np.all(s > 0)


def test_coverage():
    c = coverage(np.zeros(4), np.zeros(4) + [0.5, 1.5, 2.5, 0.0], np.ones(4))
    assert c["within_1sigma"] == 0.5 and c["within_2sigma"] == 0.75


def test_active_learning_loop_and_stopping():
    pool = np.linspace(-30, 30, 21)
    truth = synthetic_curve(pool)
    for strat in ("spread", "random"):
        r = active_learning(lambda x: synthetic_curve(x)[0], pool, lambda s: MLPEnsemble(n_members=2, epochs=100, seed=s),
                            n_init=3, strategy=strat, threshold=1e-9, max_steps=3, truth=truth)
        assert len(r["idx"]) == 6 and r["stopped_by"] == "max_steps" and len(set(r["idx"])) == 6
