"""Active learning on a candidate pool: start from a few points, add the pool point with the largest predicted spread
(or a random one, for the baseline), refit, and stop when the largest spread over the pool falls below a threshold
(and at least n_min points are in), or after max_steps. The 'oracle' returns the (possibly noisy) target value: here a SYNTHETIC function, later LAMMPS."""
import numpy as np


def active_learning(oracle, pool, make_model, n_init=3, strategy="spread", threshold=0.02, max_steps=15, seed=0, truth=None, n_min=5):
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool, float).reshape(len(pool), -1)
    idx = list(rng.choice(len(pool), n_init, replace=False))
    y = [float(oracle(pool[i])) for i in idx]
    hist = []
    for step in range(max_steps + 1):
        model = make_model(seed).fit(pool[idx], np.array(y))
        m, s = model.predict(pool)
        rmse = float(np.sqrt(np.mean((m - truth) ** 2))) if truth is not None else np.nan
        hist.append(dict(step=step, n=len(idx), max_spread=float(s.max()), rmse=rmse))
        if (s.max() < threshold and len(idx) >= n_min) or step == max_steps:
            break
        free = [i for i in range(len(pool)) if i not in idx]
        nxt = free[int(np.argmax(s[free]))] if strategy == "spread" else int(rng.choice(free))
        idx.append(nxt); y.append(float(oracle(pool[nxt])))
    return dict(hist=hist, idx=idx, y=y, model=model, stopped_by="threshold" if hist[-1]["max_spread"] < threshold else "max_steps")


def compare_strategies(oracle, pool, make_model, truth, seeds=(0, 1, 2), **kw):
    """Run 'spread' and 'random' selection for several seeds; return rmse after each added point."""
    out = {}
    for strat in ("spread", "random"):
        out[strat] = [active_learning(oracle, pool, make_model, strategy=strat, seed=s, truth=truth, **kw)["hist"] for s in seeds]
    return out
