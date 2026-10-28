"""Ensemble of small PyTorch MLPs: the prediction is the ensemble mean and its spread (standard deviation across members).

Diversity between members (a plain ensemble of identically trained nets was badly overconfident on the synthetic test:
6.6 % of true values inside +-1 spread, Session 10):
  - each member has its own seed and trains on its own bootstrap resample of the data (bagging);
  - each member adds a fixed, untrained random "prior" network times prior_scale (randomized prior functions,
    Osband, Aslanides & Cassirer, NeurIPS 2018), so members disagree away from the data.
"""
import numpy as np
import torch


class MLPEnsemble:
    def __init__(self, n_members=5, hidden=(32, 32), epochs=2000, lr=1e-2, weight_decay=0.0, seed=0, prior_scale=1.0, bootstrap=True,
                 noise_std=None):
        self.n_members, self.hidden, self.epochs, self.lr, self.wd, self.seed = n_members, hidden, epochs, lr, weight_decay, seed
        self.prior_scale, self.bootstrap = prior_scale, bootstrap
        self.noise_std = noise_std     # measurement noise of the targets (e.g. the Peierls-bracket uncertainty, GPa)
        self.members = []

    def _net(self, d):
        layers, n_in = [], d
        for h in self.hidden:
            layers += [torch.nn.Linear(n_in, h), torch.nn.Tanh()]
            n_in = h
        layers.append(torch.nn.Linear(n_in, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float)).reshape(len(y), -1); y = np.asarray(y, float)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-12          # standardise inputs and target
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        yt = torch.tensor(((y - self.ym) / self.ys)[:, None], dtype=torch.float32)
        self.members = []
        for k in range(self.n_members):
            torch.manual_seed(self.seed * 1000 + k)
            rng = np.random.default_rng(self.seed * 1000 + k)
            net, prior = self._net(X.shape[1]), self._net(X.shape[1])
            for p in prior.parameters():
                p.requires_grad_(False)
            b = rng.integers(0, len(y), len(y)) if self.bootstrap else np.arange(len(y))
            Xb, yb = Xt[b], yt[b]
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = torch.mean((net(Xb) + self.prior_scale * prior(Xb) - yb) ** 2)
                loss.backward(); opt.step()
            self.members.append((net, prior))
        return self

    def predict(self, X, include_noise=False):
        """Mean and spread; include_noise=True adds noise_std in quadrature (spread of a new measurement)."""
        X = np.atleast_2d(np.asarray(X, float))
        X = X.reshape(-1, len(self.xm))
        Xt = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32)
        with torch.no_grad():
            P = np.stack([(n(Xt) + self.prior_scale * p(Xt)).numpy()[:, 0] for n, p in self.members]) * self.ys + self.ym
        m, s = P.mean(0), P.std(0)
        if include_noise and self.noise_std:
            s = np.sqrt(s ** 2 + self.noise_std ** 2)
        return m, s
