"""Event-aware surrogates for v0.2. EventGPC (end of file) is the model in use: Q10 revised, V02_PLAN section 8.2 (a GP
classifier picks the event). EventGP below is the original section 8 Q10 model, kept for the comparison and the record: its
split GP can undercut between separate patches of split data (diagnosis, commit 44d2532).

EventGP — Event-aware two-GP surrogate for v0.2 (Ansh, V02_PLAN section 8 Q10, 2 Nov 2026); settings fixed before the synthetic study.

Two GP surfaces over x = (chi / 30, sigma_yz / s): tau_glide and tau_split. The measured tau_P is the lower of the two, and
the D5 event type says which one was observed; each GP is fitted ONLY on the points of its own event type.
Prediction = the lower of the two posterior means. Spread = standard deviation of min(g, s) over N_SAMPLES draws from the
two posterior marginals (independent per point), used by active learning.
Prior means (constant, subtracted before fitting):
  split GP: the HIGHEST observed tau_P (all event types), so where it has no data it sits above every observed value and
            cannot undercut the glide surface (Q10);
  glide GP: the mean of the glide points (as the v0.1 GPs, normalize_y); if there are no glide points yet, the highest
            observed tau_P as well.
Kernel per GP (as v0.1 D4, in 2D): Constant(1, [1e-3, 1e3]) x RBF(length scales [0.5, 0.5], bounds [1e-2, 1e2]) + White, with
the white-noise std learned within [0.0029, 0.1] GPa (D4). Targets are divided by a fixed scale = max(std of all tau_P, 0.05 GPa).
A type with fewer than MIN_FIT = 3 points is not optimised: its kernel keeps the initial hyperparameters (amplitude 1 in
scaled units, length scale 0.5, noise 0.0029 GPa). A type with no points predicts its prior mean with the prior std.
Optimiser restarts: N_RESTARTS = 2, random_state = seed.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

NOISE_BOUNDS = (0.0029, 0.1)          # GPa, D4
MIN_FIT, N_RESTARTS, N_SAMPLES = 3, 2, 256
CHI_SCALE = 30.0


def scale_x(chi, sig, s=1.0):
    return np.column_stack([np.asarray(chi, float) / CHI_SCALE, np.asarray(sig, float) / s])


class _OneGP:
    def __init__(self, prior_mean, yscale, seed):
        self.m0, self.ys, self.seed, self.gp = prior_mean, yscale, seed, None

    def fit(self, X, y):
        lo, hi = (b / self.ys for b in NOISE_BOUNDS)
        k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF([0.5, 0.5], (1e-2, 1e2))
             + WhiteKernel(lo ** 2, (lo ** 2, hi ** 2)))
        if len(y) == 0:
            self.prior_std = float(np.sqrt(1.0 + lo ** 2) * self.ys)
            return self
        opt = len(y) >= MIN_FIT
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=False, optimizer="fmin_l_bfgs_b" if opt else None,
                                           n_restarts_optimizer=N_RESTARTS if opt else 0, random_state=self.seed)
        self.gp.fit(X, (np.asarray(y, float) - self.m0) / self.ys)
        return self

    def predict(self, X):
        if self.gp is None:
            return np.full(len(X), self.m0), np.full(len(X), self.prior_std)
        m, s = self.gp.predict(X, return_std=True)
        return self.m0 + m * self.ys, s * self.ys


class EventGP:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y, event):
        X, y, event = np.asarray(X, float), np.asarray(y, float), np.asarray(event)
        ys = max(float(np.std(y)), 0.05)
        top = float(np.max(y))
        g, s = event == "glide", event == "split"
        self.glide = _OneGP(float(np.mean(y[g])) if g.any() else top, ys, self.seed).fit(X[g], y[g])
        self.split = _OneGP(top, ys, self.seed).fit(X[s], y[s])
        self.n_glide, self.n_split = int(g.sum()), int(s.sum())
        return self

    def predict_both(self, X):
        X = np.asarray(X, float)
        return self.glide.predict(X), self.split.predict(X)

    def predict(self, X, rng=None):
        """(mean, spread, predicted event). mean = lower posterior mean; spread from N_SAMPLES draws of min(g, s)."""
        (mg, sg), (ms, ss) = self.predict_both(X)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        zg = rng.standard_normal((N_SAMPLES, len(mg))); zs = rng.standard_normal((N_SAMPLES, len(mg)))
        spread = np.minimum(mg + sg * zg, ms + ss * zs).std(axis=0)
        return np.minimum(mg, ms), spread, np.where(mg <= ms, "glide", "split")


class EventGPC:
    """Q10 revised (Ansh, fix (c), V02_PLAN section 8.2, 2 Nov 2026): a GP classifier decides the event type; the two regression
    GPs (as in EventGP, split prior mean = highest observed tau_P) give tau_P. p = P(split | x); predicted event = split if
    p > 0.5; predicted tau_P = that event's GP mean. AL spread = std over N_SAMPLES draws (event ~ Bernoulli(p), tau ~ that
    GP's posterior marginal). 0 split points picked: classifier skipped, p = 0. 0 glide points: classifier skipped, p = 1.
    Classifier: GaussianProcessClassifier, Constant(1, [1e-2, 1e2]) x RBF([0.5, 0.5], [1e-2, 1e2]), Laplace, 2 restarts,
    max_iter_predict 100, random_state = seed."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y, event):
        from sklearn.gaussian_process import GaussianProcessClassifier
        X, y, event = np.asarray(X, float), np.asarray(y, float), np.asarray(event)
        self.reg = EventGP(self.seed).fit(X, y, event)
        self.n_glide, self.n_split = self.reg.n_glide, self.reg.n_split
        self.glide, self.split = self.reg.glide, self.reg.split
        lab = (event == "split").astype(int)
        if lab.min() == lab.max():
            self.clf, self.p_const = None, float(lab[0])
        else:
            k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF([0.5, 0.5], (1e-2, 1e2))
            self.clf = GaussianProcessClassifier(kernel=k, n_restarts_optimizer=2, max_iter_predict=100,
                                                 random_state=self.seed).fit(X, lab)
        return self

    def p_split(self, X):
        X = np.asarray(X, float)
        return np.full(len(X), self.p_const) if self.clf is None else self.clf.predict_proba(X)[:, 1]

    def predict_both(self, X):
        return self.reg.predict_both(X)

    def predict(self, X, rng=None):
        """(mean, spread, predicted event): event = split if P(split) > 0.5; mean = that event's GP mean."""
        X = np.asarray(X, float)
        (mg, sg), (ms, ss) = self.predict_both(X)
        p = self.p_split(X)
        is_split = p > 0.5
        rng = np.random.default_rng(self.seed) if rng is None else rng
        draw = rng.random((N_SAMPLES, len(X))) < p
        z = rng.standard_normal((N_SAMPLES, len(X)))
        samples = np.where(draw, ms + ss * z, mg + sg * z)
        return np.where(is_split, ms, mg), samples.std(axis=0), np.where(is_split, "split", "glide")


class EventGPD(EventGPC):
    """Fix (d) (Ansh, V02_PLAN section 8.3, 2 Nov 2026): EventGPC's classifier and regression GPs (same settings), with the
    min rule restored inside the classifier's split region. p = P(split | x) <= 0.5: glide, tau_P = glide GP mean.
    p > 0.5: tau_P = min(glide mean, split mean), event = the lower one. AL spread: std over N_SAMPLES draws; event branch
    ~ Bernoulli(p); glide branch: a glide-GP sample; split branch: min(glide sample, split sample)."""

    def predict(self, X, rng=None):
        X = np.asarray(X, float)
        (mg, sg), (ms, ss) = self.predict_both(X)
        p = self.p_split(X)
        in_split = p > 0.5
        split_lower = ms < mg
        event = np.where(in_split & split_lower, "split", "glide")
        mean = np.where(in_split, np.minimum(mg, ms), mg)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        draw = rng.random((N_SAMPLES, len(X))) < p
        g = mg + sg * rng.standard_normal((N_SAMPLES, len(X)))
        s = ms + ss * rng.standard_normal((N_SAMPLES, len(X)))
        samples = np.where(draw, np.minimum(g, s), g)
        return mean, samples.std(axis=0), event
