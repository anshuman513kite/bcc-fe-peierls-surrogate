"""v0.2 synthetic pool-size study (Ansh's D-v02-2; measure fixed in V02_PLAN section 8 before this ran). SYNTHETIC data only.

For each synthetic surface (src/surrogate/synthetic2d.py: weak / medium / strong sigma_yz effect):
  1. N = median over 20 seeds of the picks RANDOM selection needs to reach RMSE <= 0.02 GPa on the synthetic T (7 x 7),
     picking from a large pool (make_pool(2000)); a seed that has not reached it after N_CAP picks counts as N_CAP + 1.
     N is rounded up to an integer.
  2. For pool sizes k N (k = 3, 5, 8; make_pool(k N), disjoint from T, with chi = +-30 edge points), the picks needed by
     active learning (largest spread, seeded random start of N_INIT points), random, nested regular grid (deterministic, one
     run) and space-filling (seeded greedy maximin) for seeds 0..19. A run that never reaches the target counts as pool size + 1.
  3. Advantage(k) = median over seeds of (random_k - AL_k); 90 % bootstrap interval of that median (2000 resamples, seed BOOT_SEED).
     Stops growing between k and the next k': advantage(k') is strictly below the upper end of the interval at k (inside
     it, or below it: the advantage did not grow). Chosen size = the smallest k where it stops growing (a value exactly on
     the upper end counts as growing: borderline -> the larger size, D-v02-2). Still growing at 8 -> reported, no choice.
  4. Overall size = the largest chosen size over the surfaces; nights = ceil(size / 50).
Every model is EventGPC (V02_PLAN 8.2: GP event classifier + two regression GPs; it replaced the section 8 Q10 EventGP), refitted after every pick; RMSE uses its prediction on T against the quantised
synthetic truth. Two-GP diagnostics on T at each run's final model: points where the predicted event is wrong, and
"split undercuts" (split predicted where the truth is glide) with their distance to the nearest split training point.
Usage (ml env): python v02_pool_study.py time   -> times seed 0 and prints the estimate for the full study
                python v02_pool_study.py run    -> full study; writes data/synthetic_v02/ and figures/synthetic/v02_pool_size.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import json, math, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from surrogate.event_gp import EventGP, EventGPC, EventGPD
from surrogate.designs import grid_T, make_pool, scaled, random_order, spacefill_order, grid_designs
from surrogate.synthetic2d import truth, ALPHA0

MODEL = EventGPD          # V02_PLAN 8.3 model, passed P1' and P2 (8.4, seeds 120-129); EventGPC (8.2), EventGP (Q10) kept for the record
TARGET, N_INIT, SEEDS, LARGE_POOL, N_CAP = 0.02, 3, list(range(20)), 2000, 400
SIZES, BOOT, BOOT_SEED, WORKERS, PER_NIGHT = (3, 5, 8), 2000, 20261102, 7, 50
VERSIONS = list(ALPHA0)
OUT = os.path.join(ROOT, "data", "synthetic_v02")

T = grid_T()
XT = scaled(T)


def setup(version, M):
    P, nc, ns = make_pool(M)
    y, ev = truth(P[:, 0], P[:, 1], version)
    yT, evT = truth(T[:, 0], T[:, 1], version)
    return P, nc, ns, y, ev, yT, evT


def fit_eval(P, y, ev, idx, yT, evT, seed):
    m = MODEL(seed).fit(scaled(P[idx]), y[idx], ev[idx])
    mu, _, pev = m.predict(XT)
    return m, float(np.sqrt(np.mean((mu - yT) ** 2))), pev


def diag(m, P, ev, idx, pev, evT):
    wrong_split = (pev == "split") & (evT == "glide")
    Xs = scaled(P[[i for i in idx if ev[i] == "split"]])
    dist = [float(np.min(np.linalg.norm(Xs - XT[k], axis=1))) if len(Xs) else None for k in np.where(wrong_split)[0]]
    return dict(event_wrong=int((pev != evT).sum()), split_undercut=int(wrong_split.sum()), undercut_dist=dist,
                n_split_train=m.n_split, n_glide_train=m.n_glide)


def run_order(order, P, y, ev, yT, evT, seed, cap):
    hist = []
    for n in range(N_INIT, min(len(order), cap) + 1):
        m, r, pev = fit_eval(P, y, ev, order[:n], yT, evT, seed); hist.append(r)
        if r <= TARGET:
            return n, hist, diag(m, P, ev, order[:n], pev, evT)
    return min(len(order), cap) + 1, hist, diag(m, P, ev, order[:n], pev, evT)


def run_al(P, y, ev, yT, evT, seed):
    idx, hist = random_order(len(P), seed)[:N_INIT], []
    rng = np.random.default_rng(seed + 1000)
    XP = scaled(P)
    while True:
        m, r, pev = fit_eval(P, y, ev, idx, yT, evT, seed); hist.append(r)
        if r <= TARGET:
            return len(idx), hist, diag(m, P, ev, idx, pev, evT)
        free = np.setdiff1d(np.arange(len(P)), idx)
        if len(free) == 0:
            return len(P) + 1, hist, diag(m, P, ev, idx, pev, evT)
        _, s, _ = m.predict(XP[free], rng=rng)
        idx = idx + [int(free[int(np.argmax(s))])]


def run_grid(P, nc, ns, y, ev, yT, evT):
    hist = []
    for d in grid_designs(nc, ns):
        m, r, pev = fit_eval(P, y, ev, d, yT, evT, 0); hist.append((len(d), r))
        if r <= TARGET:
            return len(d), hist, diag(m, P, ev, d, pev, evT)
    return len(P) + 1, hist, diag(m, P, ev, d, pev, evT)


def task(t):
    version, M, method, seed = t
    t0 = time.time()
    P, nc, ns, y, ev, yT, evT = setup(version, M)
    if method == "random":
        n, hist, dg = run_order(random_order(len(P), seed), P, y, ev, yT, evT, seed, N_CAP if M == LARGE_POOL else len(P))
    elif method == "spacefill":
        n, hist, dg = run_order(spacefill_order(P, seed), P, y, ev, yT, evT, seed, len(P))
    elif method == "al":
        n, hist, dg = run_al(P, y, ev, yT, evT, seed)
    else:
        n, hist, dg = run_grid(P, nc, ns, y, ev, yT, evT)
    return dict(version=version, M=M, pool=len(P), n_chi=nc, n_sig=ns, method=method, seed=seed, picks=n, hist=hist,
                seconds=round(time.time() - t0, 2), **dg)


def boot_ci(d, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    meds = np.median(d[rng.integers(0, len(d), (BOOT, len(d)))], axis=1)
    return float(np.percentile(meds, 5)), float(np.percentile(meds, 95))


def choose(adv):
    """adv: {k: (median, lo, hi)} for k in SIZES. Returns (chosen k or None, reasons)."""
    why = []
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        grows = adv[b][0] >= adv[a][2]          # on the upper end counts as growing (borderline -> larger)
        why.append(f"{a}N -> {b}N: median {adv[b][0]:+.1f} vs interval at {a}N [{adv[a][1]:+.1f}, {adv[a][2]:+.1f}]: "
                   f"{'grows' if grows else 'stops growing'}")
        if not grows:
            return a, why
    return None, why


def main(mode):
    os.makedirs(OUT, exist_ok=True)
    if mode == "time":
        t0 = time.time(); tim = {}
        for v in VERSIONS:
            r = task((v, LARGE_POOL, "random", 0)); N0 = r["picks"]
            per = {"N_random_seed0": r["seconds"]}
            for k in SIZES:
                for meth in ("al", "random", "spacefill", "grid"):
                    per[f"{meth}_{k}"] = task((v, k * N0, meth, 0))["seconds"]
            tim[v] = dict(N0=N0, **per)
            print(v, json.dumps(tim[v]), flush=True)
        seq = time.time() - t0
        grid_t = sum(tim[v][f"grid_{k}"] for v in VERSIONS for k in SIZES)
        est = ((seq - grid_t) * len(SEEDS) + grid_t) / WORKERS
        print(f"seed 0 sequential: {seq:.0f} s; estimate for the full study on {WORKERS} workers: {est / 60:.0f} min "
              f"(20 seeds x non-grid tasks + one grid run per size, / {WORKERS}; assumes seed 0 is typical)")
        json.dump(dict(timing_seed0=tim, sequential_s=seq, estimate_min=est / 60), open(os.path.join(OUT, "timing_seed0.json"), "w"), indent=1)
        return
    t0 = time.time()
    res = {"settings": dict(TARGET=TARGET, N_INIT=N_INIT, seeds=SEEDS, LARGE_POOL=LARGE_POOL, N_CAP=N_CAP, SIZES=SIZES, BOOT=BOOT,
                            BOOT_SEED=BOOT_SEED, PER_NIGHT=PER_NIGHT), "surfaces": {}}
    runs = []
    with Pool(WORKERS) as pool:
        nr = pool.map(task, [(v, LARGE_POOL, "random", s) for v in VERSIONS for s in SEEDS], chunksize=1)
        runs += nr
        N = {v: int(math.ceil(np.median([r["picks"] for r in nr if r["version"] == v]))) for v in VERSIONS}
        print(f"{time.time() - t0:.0f} s: N = {N}", flush=True)
        tasks = [(v, k * N[v], m, s) for v in VERSIONS for k in SIZES for m in ("al", "random", "spacefill") for s in SEEDS]
        tasks += [(v, k * N[v], "grid", 0) for v in VERSIONS for k in SIZES]
        runs += pool.map(task, tasks, chunksize=1)
    print(f"{time.time() - t0:.0f} s: all runs done", flush=True)
    sizes_chosen = {}
    for v in VERSIONS:
        nr = [r for r in runs if r["version"] == v and r["M"] == LARGE_POOL]
        S = dict(N=N[v], N_random_picks=[r["picks"] for r in sorted(nr, key=lambda r: r["seed"])],
                 N_not_reached=sum(r["picks"] > N_CAP for r in nr), per_size={})
        adv = {}
        for k in SIZES:
            sel = lambda m: sorted([r for r in runs if r["version"] == v and r["M"] == k * N[v] and r["method"] == m], key=lambda r: r["seed"])
            al, rnd, sf, gr = sel("al"), sel("random"), sel("spacefill"), sel("grid")[0]
            d = [b["picks"] - a["picks"] for a, b in zip(al, rnd)]
            lo, hi = boot_ci(d)
            adv[k] = (float(np.median(d)), lo, hi)
            best_other = [min(b["picks"], c["picks"], gr["picks"]) for b, c in zip(rnd, sf)]
            S["per_size"][str(k)] = dict(
                pool=al[0]["pool"], n_chi=al[0]["n_chi"], n_sig=al[0]["n_sig"],
                picks=dict(al=[r["picks"] for r in al], random=[r["picks"] for r in rnd], spacefill=[r["picks"] for r in sf], grid=gr["picks"]),
                median=dict(al=float(np.median([r["picks"] for r in al])), random=float(np.median([r["picks"] for r in rnd])),
                            spacefill=float(np.median([r["picks"] for r in sf])), grid=gr["picks"]),
                not_reached=dict(al=sum(r["picks"] > r["pool"] for r in al), random=sum(r["picks"] > r["pool"] for r in rnd),
                                 spacefill=sum(r["picks"] > r["pool"] for r in sf), grid=int(gr["picks"] > gr["pool"])),
                advantage=dict(median=adv[k][0], ci90=[lo, hi], per_seed=d),
                al_wins_strict=int(sum(a["picks"] < b for a, b in zip(al, best_other))),
                two_gp=dict(runs=len(al) + len(rnd) + len(sf) + 1,
                            runs_with_split_undercut=sum(r["split_undercut"] > 0 for r in al + rnd + sf + [gr]),
                            split_undercut_points=sum(r["split_undercut"] for r in al + rnd + sf + [gr]),
                            undercut_without_split_data=sum(r["split_undercut"] > 0 and r["n_split_train"] == 0 for r in al + rnd + sf + [gr]),
                            undercut_dist=[x for r in al + rnd + sf + [gr] for x in r["undercut_dist"]],
                            event_wrong_points=sum(r["event_wrong"] for r in al + rnd + sf + [gr])))
        k_ch, why = choose(adv)
        S["chosen_k"], S["why"] = k_ch, why
        S["chosen_size"] = None if k_ch is None else S["per_size"][str(k_ch)]["pool"]
        sizes_chosen[v] = S["chosen_size"]
        res["surfaces"][v] = S
    still = [v for v in VERSIONS if sizes_chosen[v] is None]
    res["overall"] = dict(still_growing_at_8N=still,
                          size=None if still else max(sizes_chosen.values()),
                          nights=None if still else math.ceil(max(sizes_chosen.values()) / PER_NIGHT),
                          seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(OUT, "pool_study.json"), "w"), indent=1)
    json.dump(runs, open(os.path.join(OUT, "pool_study_runs.json"), "w"))
    plot(res)
    print(json.dumps(res["overall"]))


def plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for j, (v, S) in enumerate(res["surfaces"].items()):
        x = np.array(SIZES) + (j - 1) * 0.12
        m = [S["per_size"][str(k)]["advantage"]["median"] for k in SIZES]
        lo = [S["per_size"][str(k)]["advantage"]["ci90"][0] for k in SIZES]; hi = [S["per_size"][str(k)]["advantage"]["ci90"][1] for k in SIZES]
        ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], marker="o", capsize=4,
                    label=f"{v} (N = {S['N']}; chosen {S['chosen_k'] or 'none'}N)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{k}N" for k in SIZES])
    ax.set_xlabel("candidate pool size (multiple of N, the picks random needs)")
    ax.set_ylabel("median advantage of AL over random (picks)")
    ax.set_title("SYNTHETIC v0.2 pool-size study (D-v02-2), 20 seeds, 90 % bootstrap intervals", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures", "synthetic"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "synthetic", "v02_pool_size.png"), dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
