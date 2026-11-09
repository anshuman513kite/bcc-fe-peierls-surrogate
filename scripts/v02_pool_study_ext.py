"""Extension of the v0.2 pool-size study (V02_PLAN section 8.5, Ansh, 2 Nov 2026). SYNTHETIC data only, no LAMMPS.
STRONG surface only, pool sizes 12N and 16N (N = 47 from the study of record, commit 49bb451), seeds 0-19, methods AL,
random, space-filling (20 seeds each) and grid (deterministic, one run), model EventGPD. Reuses task(), boot_ci() and
choose() of v02_pool_study.py unchanged; choose() is run over (3, 5, 8, 12, 16) by setting its module constant SIZES, so the
section 8.1 rule and the D-v02-2 edge rules are exactly the committed ones.
Steps:
  1. Reproduction check FIRST: recompute the strong advantage at 3N / 5N / 8N from the committed pool_study_runs.json and
     compare with the committed pool_study.json (median and 90 % interval, exact). Any difference -> stop.
  2. Run 12N, then 16N; a progress line after each (size, method) block; partial results after each size in
     data/synthetic_v02/pool_study_ext_partial.json.
  3. section 8.1 measure for strong over 3N..16N; chosen size (cap 16N = 752 points if still growing, section 8.5); overall size =
     largest chosen size across surfaces (weak 5N, medium 3N stand); nights = ceil(size / 50).
  4. Descriptive only: per strong cell, in how many of 20 seeds AL beat the minimum of the other methods.
Writes data/synthetic_v02/pool_study_ext.json and figures/synthetic/v02_pool_size_ext.png (the committed v02_pool_size.png is
not touched). Usage (ml env): python v02_pool_study_ext.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json, math, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v02_pool_study as ps

V, NEW_SIZES, ALL_SIZES, CAP_K = "strong", (12, 16), (3, 5, 8, 12, 16), 16
METHODS = ("al", "random", "spacefill")
OUT, ROOT = ps.OUT, ps.ROOT
t0 = time.time()


def say(msg):
    print(f"{time.time() - t0:7.0f} s: {msg}", flush=True)


def cell(runs, M):
    sel = lambda m: sorted([r for r in runs if r["version"] == V and r["M"] == M and r["method"] == m], key=lambda r: r["seed"])
    return sel("al"), sel("random"), sel("spacefill"), sel("grid")[0]


def advantage(runs, M):
    al, rnd, sf, gr = cell(runs, M)
    d = [b["picks"] - a["picks"] for a, b in zip(al, rnd)]
    lo, hi = ps.boot_ci(d)
    best_other = [min(b["picks"], c["picks"], gr["picks"]) for b, c in zip(rnd, sf)]
    wins = int(sum(a["picks"] < bo for a, bo in zip(al, best_other)))
    return (float(np.median(d)), lo, hi), d, wins, dict(al=float(np.median([r["picks"] for r in al])), random=float(np.median([r["picks"] for r in rnd])),
                                                       spacefill=float(np.median([r["picks"] for r in sf])), grid=gr["picks"]), al[0]["pool"]


def main():
    S = json.load(open(os.path.join(OUT, "pool_study.json")))
    old = json.load(open(os.path.join(OUT, "pool_study_runs.json")))
    N = S["surfaces"][V]["N"]
    say(f"extension: {V}, N = {N} (committed), new sizes {NEW_SIZES} -> pools {[k * N for k in NEW_SIZES]} (target M; actual size from make_pool)")
    # 1. reproduction check
    for k in (3, 5, 8):
        (m, lo, hi), *_ = advantage(old, k * N)
        c = S["surfaces"][V]["per_size"][str(k)]["advantage"]
        ok = m == c["median"] and lo == c["ci90"][0] and hi == c["ci90"][1]
        say(f"reproduce {k}N: median {m:+.1f} [{lo:+.1f}, {hi:+.1f}] vs committed {c['median']:+.1f} [{c['ci90'][0]:+.1f}, {c['ci90'][1]:+.1f}]: {'identical' if ok else 'DIFFERENT'}")
        if not ok:
            say("STOP: the committed 3N/5N/8N numbers are not reproduced; nothing run"); sys.exit(1)
    # 2. new runs
    new = []
    with Pool(ps.WORKERS) as pool:
        for k in NEW_SIZES:
            for m in METHODS:
                new += pool.map(ps.task, [(V, k * N, m, s) for s in ps.SEEDS], chunksize=1)
                say(f"{k}N {m}: 20 runs done")
            new += pool.map(ps.task, [(V, k * N, "grid", 0)])
            say(f"{k}N grid: 1 run done")
            json.dump(new, open(os.path.join(OUT, "pool_study_ext_partial.json"), "w"))
            say(f"{k}N: partial results saved ({len(new)} runs so far)")
    # 3. measure over all sizes
    runs = [r for r in old if r["version"] == V] + new
    adv, cells = {}, {}
    for k in ALL_SIZES:
        a, d, wins, med, pool_n = advantage(runs, k * N)
        adv[k] = a
        cells[str(k)] = dict(pool=pool_n, advantage=dict(median=a[0], ci90=[a[1], a[2]], per_seed=d), medians=med, al_wins_strict=wins)
    ps.SIZES = ALL_SIZES                                   # the committed choose() iterates over its module SIZES
    k_ch, why = ps.choose(adv)
    capped = k_ch is None
    k_final = CAP_K if capped else k_ch
    strong_size = cells[str(k_final)]["pool"]
    chosen = {"weak": S["surfaces"]["weak"]["chosen_size"], "medium": S["surfaces"]["medium"]["chosen_size"], "strong": strong_size}
    size = max(chosen.values())
    res = dict(section="V02_PLAN 8.5", N=N, cells=cells, why=why, strong_chosen_k=k_final, strong_still_growing_at_cap=capped,
               chosen_sizes=chosen, overall_size=size, nights=math.ceil(size / ps.PER_NIGHT), seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(OUT, "pool_study_ext.json"), "w"), indent=1)
    json.dump(new, open(os.path.join(OUT, "pool_study_ext_runs.json"), "w"))
    for k in ALL_SIZES:
        c = cells[str(k)]
        say(f"strong {k}N (pool {c['pool']}): advantage {c['advantage']['median']:+.1f} [{c['advantage']['ci90'][0]:+.1f}, {c['advantage']['ci90'][1]:+.1f}]; "
            f"medians {c['medians']}; AL beat min of others in {c['al_wins_strict']}/20 seeds (descriptive)")
    for w in why:
        say(w)
    say(f"strong chosen: {k_final}N ({strong_size} points){' — STILL GROWING AT THE CAP (16N)' if capped else ''}; "
        f"chosen sizes {chosen}; overall size {size}; nights {res['nights']}")
    plot(S, cells, k_final, capped)


def plot(S, cells, k_final, capped):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    series = [(v, (3, 5, 8), {str(k): S["surfaces"][v]["per_size"][str(k)] for k in (3, 5, 8)}, f"{S['surfaces'][v]['chosen_k']}N")
              for v in ("weak", "medium")]
    series.append((V, ALL_SIZES, cells, f"{k_final}N" + (", still growing at cap" if capped else "")))
    for j, (v, ks, cc, lab) in enumerate(series):
        x = np.array(ks) + (j - 1) * 0.15
        m = [cc[str(k)]["advantage"]["median"] for k in ks]
        lo = [cc[str(k)]["advantage"]["ci90"][0] for k in ks]; hi = [cc[str(k)]["advantage"]["ci90"][1] for k in ks]
        ax.errorbar(x, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], marker="o", capsize=4,
                    label=f"{v} (N = {S['surfaces'][v]['N']}; chosen {lab})")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(ALL_SIZES); ax.set_xticklabels([f"{k}N" for k in ALL_SIZES])
    ax.set_xlabel("candidate pool size (multiple of N, the picks random needs)")
    ax.set_ylabel("median advantage of AL over random (picks)")
    ax.set_title("SYNTHETIC v0.2 pool-size study + strong extension (V02_PLAN 8.5), 20 seeds, 90 % intervals", fontsize=8.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "synthetic", "v02_pool_size_ext.png"), dpi=150)


if __name__ == "__main__":
    main()
