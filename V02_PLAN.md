# Project A v0.2 — plan (written 2 Nov 2026, before any v0.2 data)

Planning only: nothing in this file has been run. Every timing below is an **estimate built from measured v0.1 runs**
(the run is named each time); every physics setting marked **[Ansh]** is a proposal awaiting Ansh's decision.
Machine: 7.6 GB RAM, 7 MPI ranks (NP = cores − 1), one LAMMPS run at a time, every run through `tools/launch_safe.sh`.

---

## 1. The pre-registered design (copied word for word from README.md, written 28 Oct 2026)

> ## Pre-registered v0.2 test (written 28 Oct 2026, before any v0.2 data)
> - **Inputs:** the MRSS angle χ in [−30°, +30°] and a second, non-glide stress σ_yz. The σ_yz range is set only after its
>   own loading gate (a perfect-slab calibration like Gate A) passes.
> - **Ground truth:** a dense grid of LAMMPS runs over (χ, σ_yz), held out from every method.
> - **Metric:** the number of LAMMPS runs a method needs to reach RMSE ≤ 0.02 GPa on the held-out grid.
> - **Contenders:** spread-based active learning, random choice, a regular grid, and a space-filling design; 20 seeds for each
>   method with randomness.
> - **Claim rule:** a saving is claimed only if active learning needs fewer runs than the best other method in at least 15 of
>   20 seeds. Otherwise this README will report that active learning did not help.
> - **Model:** the surrogate will know the event type (glide jump vs easy → split), since one smooth model cannot fit both.

Nothing below changes this text. Section 6 lists the points it leaves open; any answer is added as a dated addendum
**before** the first dense-grid run, never after.

---

## 2. What σ_yz is (plain words)

In our frame the dislocation line and b are along x = [111], the glide plane normal is y = [−110], the glide direction is
z = [−1−12]. σ_yz is a shear on the (−110) glide plane that pushes **across** the Burgers vector (along z), not along it.
On a straight screw dislocation it exerts **no Peach–Koehler force**: F = (σ·b) × ξ = b (0, σ_xz, −σ_xy) for b and ξ along x,
so σ_yz does not appear. Any change of the Peierls stress it causes is a non-Schmid effect that acts through the core
structure. (How it maps onto the "non-glide stress" of Gröger, Bailey & Vitek 2008 has **not been checked**.)

---

## 3. Measured v0.1 timings and memory used for the estimates

All small-cell figures are the 146.87 Å cell (27,972 atoms), sweep protocol (option-2 loading, CG etol 0, ftol 1e-6,
dmax 0.01 Å, 0–2.4 GPa in 50 MPa, then 10 MPa), 7 ranks.

| v0.1 run (log) | What it did | Measured wall time |
|---|---|---|
| `logs/chi_small_m30.log` | χ = −30 ramp, 27 steps | 377 s (14.0 s/step) |
| `logs/chi_small_m18.log`, `_m12`, `_m6` | χ = −18, −12, −6 ramps | 385, 423, 470 s |
| `logs/chi_small_0.log` | χ = 0 ramp (B2), 30 steps | 480 s (16.0 s/step) |
| `logs/chi_small_p6.log`, `_p12`, `_p18`, `_p24` | χ = +6 … +24 ramps | 537, 489, 523, 603 s |
| `logs/chi_small_p30.log` | χ = +30 ramp, 46 steps | 641 s (13.9 s/step) |
| `logs/chi_small_p28.log` | χ = +28 ramp, 46 steps (slowest small-cell ramp) | 742 s |
| `logs/al_round3.log` | one complete point, χ = −29: Gate A calibration (8 points) 12 s, ramp 384 s, 2 stability tests + record 49 s | 7 min 28 s total |
| `logs/chi_calibration.pid` → `.out` (file times) | full Gate A v2 calibration, 11 angles × 4 τ + model checks | ≈ 170 s |
| `logs/chi_x2_m30.log`, `logs/chi_x2_p30.log` | 2× cell (55,944 atoms) ramps | 464 s (28 steps), 790 s (46 steps) |
| `sims/07_al_live/chi_small_m29/stab/log.*` | one stability test, LAMMPS part | 7–8 s |

Per small-cell ramp step: 13.9–16.8 s. Per complete point as run in v0.1 (calibration 12 s + ramp + 2 stability tests 49 s):
7.3 min (χ = −30) to 13.4 min (χ = +28), depending on the Peierls stress (more coarse steps for a harder angle).

**Memory — what was measured and what was not.** LAMMPS itself reports its allocation per rank (max over ranks):
small cell 4.75 MB during a ramp step and 6.12 MB in a stability/per-atom-energy run; 2× cell 11.41 MB; Gate A perfect slab
5.28 MB (logs as above). The cell builder peaked at 35 MB (Session 6). **The total memory of the running processes (MPI,
Python, libraries) was never measured in v0.1.** The LAMMPS figures are therefore only a lower bound (7 × 6.12 MB ≈ 43 MB
small, 7 × 11.41 MB ≈ 80 MB 2×). Step 2 below adds a memory sampler to the launcher so that v0.2 reports measured peak RAM.
Nothing in v0.1 came near the 7.6 GB limit once the builder was fixed, but that statement rests on the absence of crashes,
not on a measurement.

---

## 4. Steps in order

| # | Step | LAMMPS? | Estimated wall time (source) | Peak RAM (source) |
|---|---|---|---|---|
| 0 | This plan; CLAUDE.md push rule | no | — | — |
| 1 | **Ansh decides the open points of §6** — **done, §8** | no | — | — |
| 2 | **Code, no runs:** σ_yz surface force in `tools/in.step_chi`, `tools/peierls_chi.py`, `checks/08_gateA_rebox/in.measure`, `calibrate_chi.py`; three-component Gate A in `tools/gate_a.py`; σ_yz pre-ramp; grid queue with resume and per-point status; memory sampler in `tools/launch_safe.sh` (sum of RSS of all `lmp` processes every 10 s into `logs/<run>.log`); event-aware 2D surrogate (two GPs, §8 Q10) and the four design generators; tests (synthetic 2D, criterion edge cases); **synthetic pool-size study (D-v02-2): 4 methods × 20 seeds at pools of 3×, 5× and 8× the picks needed, on synthetic 2D data only** | no | code: one working block; pytest suite (21 tests in v0.1) ~1 min; pool-size study: not measured (no v0.1 run of this size), timed on one seed first and estimated from that | small (ml env, 1 thread); not measured in v0.1 |
| 3 | **Gate A-yz** on the perfect slab (§5.3, part 1), **calibration and ramp top extended to τ = 3.0 GPa (§8 Q5)**: 3 χ × 9 σ_yz (the full ladder ±0.25 … ±1.0, plus 0 re-calibrated to 3.0) × (7 τ calibrated + 6 model checks), first with the unchanged σ_yz = 0 strain model, then calibrated if needed | yes | 27 pairs × 20–30 s ≈ **9–14 min** (13 points per pair vs 8 in v0.1: 12 s × 13/8 ≈ 20 s, from the χ = −29 calibration 12 s, `logs/al_round3.log`; more Newton iterations at high τ allowed for) | LAMMPS 5.28 MB/rank × 7 ≈ 37 MB + process overhead (not measured) |
| 4 | **Core-must-not-move gate** in the dislocation cell (§5.3, part 2): 3 χ × 2 signs × (4 σ_yz steps + 12 τ steps) = 96 relaxations, 6 stability pairs, 6 unload checks | yes | 96 × ~15 s ≈ 24 min + stability 6 × 49 s ≈ 5 min + unload ≈ 3 min → **≈ 35 min** (per-step 13.9–16.8 s from the small-cell ramps; stability from `logs/al_round3.log`) | 6.12 MB/rank × 7 ≈ 43 MB + overhead |
| 5 | **Range probes** (§5.4): full ramps at χ ∈ {−30, 0, +30} × σ_yz = ±s, with calibration and stability | yes | 2 × (8 + 10 + 13) min ≈ **62 min** (ramps 377 / 480 / 641 s at σ_yz = 0, `logs/chi_small_m30/_0/_p30.log`, + 121 s overhead each); +15 s for every extra 50 MPa if σ_yz raises τ_P | as step 4 |
| 6 | **2× cell checks:** χ = 0 with the corrected loading (the D6 test deferred from v0.1) and χ = 0 at σ_yz = ±s | yes | 3 × ~9 min ≈ **27–35 min** (2× ramps 16.6–17.2 s/step, `logs/chi_x2_*.log`; ~31 steps to 1.32 GPa) | LAMMPS 11.41 MB/rank × 7 ≈ 80 MB + overhead |
| 7 | **Freeze the addendum** (σ_yz range s, grid T, P size from the D-v02-2 synthetic result and P layout, the §8 decisions, the README sentence on what a "run" is) with a time stamp; commit | no | — | — |
| 8 | **Night 1: held-out grid T** (49 points, §4.1) | yes | **≈ 8.2 h** + margin (below) | as step 4 |
| 9 | **Nights 2 …: candidate pool P** (size 3×, 5× or 8× the picks needed, set by D-v02-2; §4.2) | yes | **≈ 10.0 min per point, ≤ 50 points per night**; 2–7 nights depending on size (table in §4.2) | as step 4 |
| 10 | **Comparison:** 4 contenders × 20 seeds, picking from P, scored on T (strict per-seed claim rule, §8 Q8); RMSE-vs-runs curves for all four methods always reported, descriptive only (§8 Q9); fit checks; figures | no | not measured in v0.1 for this size; estimate from step 2's synthetic pool-size study | ml env, 1 thread |
| 11 | NOTES, README v0.2 results (whatever the claim rule says; the README states that a "run" in the v0.2 metric is a pick from the precomputed pool, deterministic at 0 K, Gate B'), tag v0.2 — push only if a prompt's PUSH line names it | no | — | — |

Steps 3 → 4 → 5 → 6 are strictly sequential: each one's pass is the condition for the next. Steps 3–6 together
≈ 2.2–2.5 h of LAMMPS time, so they fit one day block. No step is shortened to meet a date (D-v02-1).

### 4.1 How many dense-grid points fit in one ~10-hour night, and a proposed grid

**Per point** = Gate A calibration up to τ = 3.0 GPa (~20 s: 13 points instead of v0.1's 8, scaled from the measured 12 s)
+ σ_yz pre-ramp (≤ 4 steps × ~15 s = 60 s; new in v0.2, assumed to cost a normal step) + τ ramp + 2 stability tests and
recording (49 s) = ramp + ~129 s. The higher ramp top (3.0 GPa) costs nothing extra by itself: a ramp stops at the first
move, so it only runs longer where τ_P really is higher.

At σ_yz = 0 the measured ramps average 474 s over χ = −30, −18, −12, 0, +12, +18, +30, so one point ≈ 603 s ≈ 10.0 min:
**about 60 points in 10 h with no margin.** σ_yz may raise the Peierls stress at some angles, and each extra 50 MPa costs
~15 s per point, so the plan keeps ≈ 20 % margin: **≤ 50 points per night.**

**Proposed held-out grid T: 7 χ × 7 σ_yz = 49 points.**
- χ ∈ {−30, −20, −10, 0, +10, +20, +30}°
- σ_yz ∈ {−s, −2s/3, −s/3, 0, +s/3, +2s/3, +s}, with s fixed only after the gates (§5.4)

Estimate: ramps per σ_yz level, taking the nearest measured v0.1 angle (377 + 385 + 423 + 480 + 489 + 523 + 641 = 3318 s),
× 7 levels = 6.45 h; overhead 49 × 129 s = 1.76 h; **total ≈ 8.2 h**, leaving 1.8 h — enough for an average of ~9 extra
coarse steps per point (τ_P higher by up to ~0.45 GPa on average). If step 5 shows larger increases, T is spread over two
nights rather than thinned (D-v02-1: nothing is shortened to meet a date). The σ_yz = 0 row at χ = 0 repeats B2
(1.310–1.320 GPa) and serves as a regression check of the new code.

The 6 probe points of step 5 (χ = −30, 0, +30 at σ_yz = ±s) are nodes of T. They are computed before the range is fixed
(by a mechanical rule, §5.4), go straight into T, and are never shown to any method.

### 4.2 Candidate pool P: layout and cost at 3×, 5× and 8× (§8 Q6, Q6b, D-v02-2)

**Layout.** P is disjoint from T. It has a regular part (χ values between T's columns, e.g. {±5, ±15, ±25}, at σ_yz levels
whose spacing is set with the size) and extra points at χ = ±30 at σ_yz levels that are **not** T's levels (e.g. ±s/6, ±s/2,
±5s/6), so no method has to extrapolate to the range ends. The grid contender uses nested sub-grids of the regular part
only (Q7). The exact level counts follow from the size chosen in step 2 and go into the addendum before night 1.

**Cost.** N = the number of picks a method needs, measured in the step 2 synthetic study (not known yet). Per pool point
≈ 10.0 min (§4.1; P mixes hard and easy angles like T), **≤ 50 points per 10-h night** with the 20 % margin.
Night 1 is always T (49 points). The rows use example values of N, to be replaced by the measured one:

| N (picks needed) | P at 3× | nights for P | P at 5× | nights for P | P at 8× | nights for P |
|---|---|---|---|---|---|---|
| 20 | 60 | 2 | 100 | 2 | 160 | 4 |
| 30 | 90 | 2 | 150 | 3 | 240 | 5 |
| 40 | 120 | 3 | 200 | 4 | 320 | 7 |

Nights for P = ⌈size / 50⌉; total nights = 1 (T) + nights for P. LAMMPS time without margin ≈ size × 10.0 min
(e.g. 240 points ≈ 40 h). Every night is a separate crash-safe launch with resume (step 2); RAM per run is unchanged
(one small-cell run at a time).

---

## 5. σ_yz loading design (not implemented)

### 5.1 How σ_yz is applied
**Plain words.** Exactly as σ_xy is applied now, but pushing along z instead of x: equal and opposite forces on the
2-plane slabs at the two free y surfaces, top +F along z, bottom −F along z. The opposite shear on the z faces is
carried through the periodic z boundary, as σ_xy's is carried through the periodic x boundary.

**Detail.**
- One `fix addforce` per slab with both components, F = (σ_xy, 0, σ_yz) · Lx·Lz / N_slab on the top slab and its negative on
  the bottom, with `fix_modify energy yes` (the minimiser's energy then includes the work of both components). Small cell:
  N_slab = 756, A = 2179.12 Å², 0.017991 eV/Å per atom per GPa (Session 5 hand check, same number for z).
- σ_xz stays as in v0.1: box strains (ε_xx, ε_zz, γ_xz) from the Gate A calibration, written by `tools/rebox.py`.
- **Load path (decided, §8 Q1):** σ_yz first, at τ = 0, in steps of ≤ 0.25 GPa from the relaxed cell; then the usual τ
  ramp (50 MPa, then 10 MPa) at fixed χ and fixed σ_yz. Alternative: raise σ_yz in proportion to τ. At 0 K the path
  only matters if something irreversible happens on the way; "σ_yz first" makes such an event visible on its own
  (the core-must-not-move gate checks it) and matches the question "how does a fixed non-glide stress change τ_P".

**Does it conflict with the existing set-up?** Linear-elastic calculation (rotated stiffness from our C11, C12, C44;
`checks/09_yz_coupling/yz_coupling.py`, output `output.txt`; same method as `checks/03_tilt_xz/slab_case.py`): with the y faces loaded by
σ_yz = 1 GPa and the box strains unchanged, the slab takes γ_yz = +0.01253 and γ_xy = −0.00553 and **every other stress
component is zero** (σ_xx = σ_yy = σ_zz = σ_xz = σ_xy = 0 to round-off). The C' column for γ_yz is (0, 0, 0, 93.71, 0, 31.50) GPa.
- **b/2 box tilt (xz):** no conflict. The tilt makes the cell periodic along z for the screw's x-displacement; σ_yz moves
  the top half along z relative to the bottom (γ_yz·Ly ≈ 1.9 Å per GPa, calculated), which leaves the x-periodicity and
  the tilt untouched.
- **PAD periodic glide:** no conflict in the mechanics. z is periodic, the force sums to zero (no drift), and every image
  dislocation in the PAD row feels the same uniform σ_yz, with zero Peach–Koehler force. One bookkeeping effect: the
  shear slides the core region along z by γ_yz·(y_core − y_centre) ≈ 0.0125 × (−0.7 Å) ≈ −0.01 Å per GPa (calculated),
  far below one jump (2.331 Å); D5 compares discrete core sites and is unaffected. The PAD built-in shear is a σ_xz-type
  stress (±0.69 GPa estimated, opposite in the two halves), a different component; step 6 checks the cell size again.
- **Calibrated box strains for σ_xz:** no first-order conflict (calculation above). **Second-order cross terms
  (τ·σ_yz, σ_yz²) are unknown**; v0.1 showed second-order terms of 1–3 % for σ_xy and σ_xz (Sessions 9–10). So the v0.1
  strain model (`strain_model.json`, `strain_model_al.json`) is **valid only at σ_yz = 0 until Gate A-yz shows otherwise.**
  If it fails, every (χ, σ_yz) point gets its own calibration, as every new χ did in v0.1 (12–20 s per point, measured).

### 5.2 Gate A extended to three loaded components
Measured interior stress of the perfect slab (average of stress/atom within 8 Å of mid-plane, minus the unloaded slab,
as in v0.1). A point passes when all four hold:

| Component | Target | Criterion |
|---|---|---|
| σ_xy | τ cos χ | within 1 % of target (v0.1 Gate A) |
| σ_xz | τ sin χ | within 1 % of target **OR** loading-angle error \|Δσ_xz\|/τ ≤ 0.1° = 0.001745 rad (D2) |
| σ_yz | σ_yz target | within 1 % of target **OR** an absolute floor for targets near zero **[Ansh]** — options below |
| σ_xx, σ_yy, σ_zz | 0 | each < 2 % of τ; at τ = 0 points, < 2 % of \|σ_yz\| (i.e. of the largest applied stress) **[Ansh]** |

Why a floor for σ_yz: D2 was needed because 1 % of a near-zero target (σ_xz at χ = +3°) is an extreme demand
(0.00032 GPa failed at τ = 0.5). The same happens for σ_yz = 0 and for small σ_yz levels. Options:
- **(a) D2-style (recommended):** \|Δσ_yz\| ≤ 0.001745 × max(τ, \|σ_yz\|). The error, as a fraction of the main applied stress,
  is no larger than what D2 already accepts for σ_xz. At τ = 1–2.4 GPa this is 0.0017–0.0042 GPa.
- (b) A fixed floor, e.g. \|Δσ_yz\| ≤ 0.002 GPa.

For scale (measured, v0.1): with σ_yz not loaded, the Gate A slab's leftover σ_yz was ≤ 0.00002 GPa
(`logs/al_round3.log`, χ = −29) and ≤ 0.0001 GPa at χ = 0 (Session 9). Either floor sits far above that.
The criterion, whichever is chosen, is fixed before step 3 runs.

### 5.3 The proposed gate test
**Part 1 — Gate A-yz (perfect slab, 2400 atoms, rebox path, as `checks/08_gateA_rebox/`).**
- χ ∈ {−30, 0, +30}: the two ends (largest σ_xz, both signs) and χ = 0 (σ_xz target zero, so the D2 branch is exercised).
- σ_yz ∈ {±0.25, ±0.5, ±0.75, ±1.0} GPa (the full accepted ladder, §8 Q2, since s must be one of these values), plus
  σ_yz = 0 re-calibrated to the new ramp top.
- τ ∈ {0, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0} calibrated, plus model checks at 0.5, 1.25, 1.75, 2.2, 2.55, 2.85 (the v0.1 scheme
  extended to 3.0 GPa, §8 Q5).
- Run A: the v0.1 strain model unchanged (does σ_yz disturb it?). Run B, only if A fails: Newton calibration per point.
- Result: the largest ladder value s_A at which every point passes.

**Part 2 — the core must not move (dislocation cell, 146.87 Å, only if part 1 passes).**
- χ = −30: the lowest measured Peierls stress (1.160–1.170 GPa) and its first event is easy → split, the event a
  non-glide stress is most likely to trigger. χ = +30: the largest σ_xz and glide on (−1 0 1). χ = 0: reference.
- σ_yz ladder at τ = 0: 0 → ±0.25 → ±0.5 → ±0.75 → ±1.0 GPa (up to s_A), each step relaxed from the previous one.
- Then, at the top σ_yz of each branch, τ from 0 to **0.6 GPa** in 50 MPa steps **[Ansh]**: 52 % of the lowest measured
  v0.1 bracket (1.16 GPa). The margin is wide on purpose, because nobody knows yet how much σ_yz lowers τ_P.
- Then unload (τ → 0, then σ_yz → 0) back to the starting stress.

**How "did not move" is detected** (every relaxed state; thresholds **[Ansh]**):
1. D5 state check (`tools/moved_rule.py`, `dd_state` / `same_state`): the core is still **compact** (top column < 1.4 b)
   **at the start site**. Any glide jump, easy → split transition or "anomaly" fails the point.
2. Largest non-affine atom displacement between consecutive steps ≤ 0.1 Å. Measured in v0.1: ordinary step 0.013 Å,
   real events 0.44–0.55 Å (Session 13 §8), so 0.1 Å separates them with room on both sides.
3. Stability test at the top of each branch (±0.001 Å random kick, 2 seeds, re-minimise at the same stress): returned.
4. Reversibility: after unloading, max |Δr| against the original relaxed cell < 0.01 Å and the energy back within
   10⁻⁴ eV. This catches an irreversible core change that 1–3 might miss.
5. Recorded, not a pass/fail: the three central-triangle DD values and the top column, to see how σ_yz distorts the core.

If the core changes at τ = 0 under σ_yz alone, that is a physics result (recorded), and that σ_yz level is outside
the usable range.

### 5.4 How the σ_yz range for the grid will be chosen (no number fixed yet)
A mechanical rule, written now, before any σ_yz data. s = the **largest ladder value** for which all of these hold:
1. Gate A-yz passes at every point with |σ_yz| ≤ s (§5.3 part 1).
2. The core-must-not-move gate passes at ±s at all three angles (§5.3 part 2).
3. Range probes (step 5: full ramps at χ = −30, 0, +30 and σ_yz = ±s): every bracket lies below the calibrated ramp top
   (**3.0 GPa**, §8 Q5; back to 2.4 GPa only if Gate A-yz fails between 2.4 and 3.0) minus one coarse step, every first event is a D5 glide jump or easy → split (no anomaly), and the stability
   tests return. If a probe fails, s drops one ladder step and the failed probes are repeated.
4. A feasibility check, **not** used to tune s: if |τ_P(±s) − τ_P(0)| < 0.02 GPa at all three probe angles, σ_yz has no
   measurable effect at the metric's resolution, the 2D test would be empty, and the work stops for Ansh to rethink.

The grid uses ±s, symmetric in sign, because a non-glide stress can act differently for + and − (to be measured, not
assumed). Because s is set by rule 1–3, seeing the probe values cannot be used to pick a "nicer" range.

---

## 6. Points the pre-registration leaves open (decide before step 3 / step 7) **[Ansh]** — decided 2 Nov, see §8 (questions kept as written)

- **Q1 Load path:** σ_yz first, then τ (recommended), or proportional (§5.1).
- **Q2 Gate ladder:** candidate values {0.25, 0.5, 0.75, 1.0} GPa. The top value is comparable to the Peierls stress
  (σ_yz/τ_P up to ~0.5–0.9). No literature range has been checked.
- **Q3 σ_yz floor:** option (a) or (b) (§5.2); "2 % of τ" at τ = 0 points read as 2 % of |σ_yz|.
- **Q4 Gate thresholds:** τ = 0.6 GPa, non-affine 0.1 Å, reversibility 0.01 Å / 10⁻⁴ eV (§5.3).
- **Q5 Ramp top:** v0.1 is calibrated and ramps to τ = 2.4 GPa. If σ_yz pushes τ_P above that at +30°, either s shrinks
  (rule 3) or the calibration is extended to higher τ. Recommended: let rule 3 shrink s.
- **Q6 One overnight grid or two:** found while planning. "Held out from every method" means the methods may not pick
  T's points, but a live 20-seed comparison (4 methods × 20 seeds × tens of runs) would take hundreds of LAMMPS hours.
  **Recommended:** a second, disjoint **candidate pool P** computed once; every method picks from P (a "run" is a look-up
  in P's table), and RMSE is scored on T. At 0 K the result is deterministic (Gate B': identical brackets on 7 and 8
  ranks), so a look-up gives exactly what a fresh run would. Proposed P: χ ∈ {±5, ±15, ±25} × the same 7 σ_yz levels
  = 42 points, disjoint from T, ≈ 7.1 h (ramps for −24, −18/−12, −6, +6, +12/+18, +24 sum to 2928 s per level; × 7 = 5.7 h;
  + 42 × 121 s = 1.4 h). P has no χ = ±30 points, so every method extrapolates the last 5° equally; adding ±30 at the
  σ_yz half-levels (+12 points, ≈ 2 h) is an option.
- **Q7 Grid contender:** nested regular sub-grids of P (2×2, 3×3, 3×4, …); runs counted = size of the first sub-grid that
  reaches RMSE ≤ 0.02 GPa. It is deterministic: one value, compared with every seed.
- **Q8 "Best other method":** recommended per seed: AL wins seed k if AL_k < min(random_k, space-filling_k, grid). Stricter
  than comparing with the method with the best median.
- **Q9 Not reached:** if a method never reaches 0.02 GPa within P, its count is "> |P|". AL cannot win a seed it does
  not finish. If no method reaches 0.02 GPa at all, the README reports that the target was not reached (v0.1 LOO RMSE
  was 0.03 GPa on the 8 glide jumps alone and 0.07 GPa on all points, so this is a real risk).
- **Q10 Event-aware model:** proposed form: two GP surfaces, τ_glide(χ, σ_yz) and τ_split(χ, σ_yz); the measured τ_P is
  the lower of the two, and the D5 event type says which one was observed (the other is only known to be higher).
  Prediction = the lower surface; spread from samples of both. Settings (kernels, noise bounds as D4) fixed and tested on
  synthetic 2D data before any v0.2 LAMMPS value is seen. AL starting design: the same 3 corner-plus-centre points for
  every method and seed, or a seeded random start (recommended: seeded random, as in v0.1).

---

## 7. What this plan does not do
No LAMMPS run, no code change to the simulation tools, no change to README.md or the public branch, no push.

---

## 8. Decisions (2 Nov 2026, before any v0.2 data)

Ansh's decisions on §6, recorded 2 Nov 2026, 17:00, before any v0.2 LAMMPS run or v0.2 data. They clarify the
pre-registered design of §1 and do not change its text.

- **Q1 Load path:** σ_yz first, at τ = 0, in steps ≤ 0.25 GPa; then the usual τ ramp (50 MPa, then 10 MPa) at fixed χ and σ_yz.
- **Q2 Gate ladder:** {0.25, 0.5, 0.75, 1.0} GPa accepted. s is set by the §5.4 rules only.
- **Q3 σ_yz tolerance:** option (a), D2-style: σ_yz passes if within 1 % of target OR |Δσ_yz| ≤ 0.001745 × max(τ, |σ_yz|).
  "Other components < 2 % of τ" reads as < 2 % of |σ_yz| at τ = 0 points.
- **Q4 Gate thresholds:** accepted: τ = 0.6 GPa in the core-must-not-move gate; non-affine displacement ≤ 0.1 Å;
  reversibility max |Δr| < 0.01 Å and energy within 10⁻⁴ eV.
- **Q5 Ramp top:** Gate A calibration and the ramp top are extended to **τ = 3.0 GPa** in step 3. Only if Gate A fails
  between 2.4 and 3.0 GPa does rule 3 (§5.4) shrink s instead.
- **Q6 Pool-based design:** a candidate pool P is computed once; every method picks from P; RMSE is scored on the held-out
  grid T. The README must state that a "run" in the v0.2 metric is a pick from the precomputed pool (deterministic at 0 K,
  shown by Gate B': identical brackets on 7 and 8 ranks).
- **Q6b Pool layout and size:** P is disjoint from T and includes χ = ±30 at σ_yz levels not in T. Its size is set by
  D-v02-2, not by a guessed ratio (§4.2).
- **Q7 Grid contender:** accepted: nested regular sub-grids of the regular part of P.
- **Q8 Claim rule, strict per-seed reading:** AL wins seed k only if AL_k < min(random_k, space-filling_k, grid). A saving is
  claimed only if AL wins at least 15 of 20 seeds.
- **Q9 Not reached:** accepted: a method that never reaches RMSE ≤ 0.02 GPa within P counts as "> |P|"; AL cannot win a
  seed it does not finish; if no method reaches it, the README says so. **Added:** RMSE-vs-runs curves for all four methods
  are always reported as descriptive results; no claim is drawn from them.
- **Q10 Event-aware model:** two GP surfaces, τ_glide(χ, σ_yz) and τ_split(χ, σ_yz); prediction = the lower one. Each GP is
  fitted only on points of its own event type. The split GP's prior mean is the highest observed τ_P, so it cannot
  undercut where it has no data. Tested on synthetic 2D data in step 2, before any v0.2 LAMMPS value is seen. AL start:
  seeded random, as in v0.1.
- **D-v02-1 (Ansh):** v0.2 is done to the full standard; no part of the design is shortened to meet a date. The test must
  be fair to active learning even if it costs extra compute.
- **D-v02-2 (Ansh):** the size of P is not fixed by a guessed ratio. In step 2, on synthetic 2D data only, all four methods
  (20 seeds) are run at pool sizes 3×, 5× and 8× the picks needed. **Rule, fixed now:** choose the smallest pool size beyond
  which AL's advantage over random stops growing (within the seed scatter); if borderline between two sizes, take the
  larger. The chosen size and the synthetic result are written into the addendum before night 1. Costs at 3×, 5× and 8×:
  §4.2.

### 8.1 Addendum (2 Nov 2026, committed 17:34:17 as 1435017, before the synthetic pool-size study ran): exact D-v02-2 measure, synthetic surfaces, settings

**Exact measure for D-v02-2 (Ansh, 2 Nov):**
- N = picks needed = median over 20 seeds of the picks random selection needs to reach RMSE ≤ 0.02 GPa on the synthetic T,
  picking from a very large pool (2000 points). Implementation details (Claude): the median is rounded up to an integer;
  a random run that has not reached the target after 400 picks stops and counts as 401 (fits on more points would make the
  study unaffordable; the number of such seeds is reported).
- Advantage at pool size kN = median over 20 seeds of (random_k picks − AL_k picks); a seed that never reaches the target
  counts as pool size + 1.
- "Stops growing": the median advantage at the next larger size lies inside the 90 % bootstrap interval (2000 resamples,
  fixed seed 20261102) of the median advantage at the current size. Choose the smallest size where it stops growing; if it
  is still growing at 8×, report that and stop for Ansh. Reading of the edges (Claude): a value **below** the interval also
  counts as "stopped growing" (the advantage fell); a value **exactly on** the upper end counts as growing (borderline → the
  larger size, as D-v02-2 says).
- Across the synthetic surfaces, the chosen pool size is the **largest** size chosen by any surface.

**Synthetic 2D surfaces** (`src/surrogate/synthetic2d.py`; SYNTHETIC, never mixed with LAMMPS data; χ in degrees, σ in units
of s = 1):
- glide at σ = 0: τ_g0(χ) = S0 / (cos χ + A cos(60° + χ)), S0 = 1.7754 GPa, A = 0.70 → 1.206 / 1.315 / 2.050 GPa at
  χ = −30 / 0 / +30 (computed).
- split at σ = 0: τ_s0 = τ_g0 + D(χ), D = 0.08 − 0.126 exp(−((χ + 30)/10)²) − 0.15 exp(−((χ − 27)/3)²) GPa → τ_s0(−30) =
  1.160 GPa (computed); split is the lower surface for χ ≲ −23 and 25 ≲ χ ≲ 29 (as −30…−24 and +28 in v0.1).
- σ effect: F = 1 + α(χ) (σ + 0.5 σ²) / 1.5 with α(χ) = α0 (0.75 + χ/120); τ_g = τ_g0 F; τ_s = τ_s0 F (1 − α0 σ / 6);
  α0 = 0.05 (weak), 0.15 (medium), 0.30 (strong). Asymmetric in the sign of σ; the split region grows for σ > 0.
  Largest |τ_P change| at |σ| = s, computed on a 0.1° grid: weak 5.0 % (σ = +1) / 1.7 % (σ = −1); medium 14.2 % / 5.0 %;
  strong 26.9 % / 10.0 %.
- τ_P = min(τ_g, τ_s), event = the lower surface; every value quantised to the 10 MPa bracket midpoint,
  floor(τ_P / 0.01) × 0.01 + 0.005.
- Synthetic T = the 7 × 7 layout of §4.1 (χ = −30, −20, …, +30; σ = 0, ±1/3, ±2/3, ±1).
- Pools (`src/surrogate/designs.py make_pool(M)`): a regular n_χ × n_σ part at cell centres plus χ = ±30 at every pool σ level;
  n_σ even, so no pool σ level equals a T level and every pool is disjoint from T; (n_χ + 2) n_σ ≈ M.

**Study settings** (fixed before running; `scripts/v02_pool_study.py`):
- Model: the two-GP EventGP of Q10 (`src/surrogate/event_gp.py`): kernel per GP Constant × RBF (2 length scales) + White, noise
  learned in [0.0029, 0.1] GPa (D4); split prior mean = highest observed τ_P; glide prior mean = mean of glide points; a GP
  with < 3 points keeps its initial hyperparameters; 2 optimiser restarts. Refitted after every pick; RMSE on T against the
  quantised truth.
- Seeds 0–19. AL: seeded random start of 3 points (the first 3 of the same seed's random order), then the largest spread
  (std of min(g, s) over 256 draws). Random: seeded permutation. Space-filling: seeded greedy maximin. Grid: nested regular
  sub-grids of the regular part, from 2 × 2 upwards, deterministic (one run per pool size, compared with every seed).
- Two-GP check: on T at each run's final model, the T points where split is predicted but the truth is glide ("split
  undercut"), and their distance to the nearest split training point.
- Run on 7 worker processes, 1 thread each (no LAMMPS runs at the same time). Seed 0 is timed first; if the full study
  would take more than 60 min it is not run (Ansh's instruction).

**Gate A at τ = 0 (Claude's reading, for Ansh to confirm):** at τ = 0 points the σ_xy and σ_xz targets are zero and the 1 % /
D2 forms are undefined; `tools/gate_a.py gate_a3_check` then holds them to the same floor as σ_yz, |Δσ| ≤ 0.001745 × |σ_yz|.

### 8.2 Addendum (2 Nov 2026, before the new code): Q10 revised — event classifier (Ansh's decision, fix (c))

**Diagnosis** (commit `44d2532`, NOTES Session 17): with the §8 Q10 model the split GP, fitted on two separate patches of
split points (near χ = −30 and near χ = +27), learned a χ length scale of 17°–43° and fitted one smooth surface through both
patches. Between them it predicted values below the glide surface, so "the lower of the two" wrongly chose split there (weak
seed 2, pick 25: 42 of 42 glide points of T, RMSE 0.195 GPa). The prior-mean safeguard (split prior = highest observed τ_P)
only protects points far from split data, and "far" is measured in the GP's own fitted length scale. Picks needed were
therefore very noisy (weak 54–102, medium 23–67 over seeds 0–2). This is a modelling effect, not a code bug.

**Decision (Ansh, 2 Nov 2026):** replace the Q10 model with fix (c), before any v0.2 LAMMPS data.

**New model (Q10 revised), settings fixed here before testing:**
- **Event classifier:** scikit-learn `GaussianProcessClassifier` on x = (χ/30, σ/s), trained on the event labels of ALL picked
  points (glide = 0, split = 1). Kernel Constant(1.0, [1e-2, 1e2]) × RBF(length scales [0.5, 0.5], bounds [1e-2, 1e2]),
  Laplace approximation (scikit-learn default), `n_restarts_optimizer = 2`, `max_iter_predict = 100`, `random_state = seed`.
- **Two regression GPs as before:** glide GP on glide points, split GP on split points, same kernels, noise bounds and
  small-sample rule as §8.1; split prior mean = highest observed τ_P (kept).
- **Prediction:** p = P(split | x). Predicted event = split if p > 0.5, else glide. Predicted τ_P = the posterior mean of
  that event's GP.
- **AL spread:** standard deviation over 256 samples. In each sample the event is drawn from Bernoulli(p), and τ from that
  event's GP posterior marginal (normal).
- **With 0 split points picked:** classifier skipped, p = 0, prediction = glide (as before). **With 0 glide points picked**
  (all split; Claude's addition by symmetry, not in Ansh's text): classifier skipped, p = 1, prediction = split.
- The old model is kept in the code as `EventGP` (for the comparison and the record); the new one is `EventGPC`. The study
  script uses `EventGPC`; all other §8.1 settings are unchanged.

**Pass criteria (fixed here; Ansh):** random selection on the 2000-point pool, FRESH seeds 100–109 (not 0–2), all three
surfaces (30 runs), old model and new model on the same seeds and the same random orders.
- **P1.** At the final pick, wrong-event predictions on T ≤ 1 point, in at least 27 of 30 runs (new model).
- **P2.** After pick 30, no RMSE rise larger than 0.05 GPa between consecutive 10-pick checks, in at least 27 of 30 runs
  (new model).
- **P3.** Median picks needed per surface, old vs new (descriptive, not a pass condition).
- **Run protocol (Claude, so that P2 is not satisfied trivially by runs that stop early):** every run continues to
  **100 picks** even after reaching RMSE ≤ 0.02 GPa (or further, up to 400, while it has not yet reached the target).
  "Picks needed" = the first pick with RMSE ≤ 0.02 GPa (401 if never within 400). "Final pick" for P1 = the last pick of
  the run. RMSE checks for P2 at picks 30, 40, 50, … up to the last multiple of 10.
- **Tests:** all 41 existing tests keep passing; new tests for the classifier path: 0 split points → glide everywhere;
  two separate split patches → no split predicted between them on a synthetic case.
- **If P1 and P2 pass:** commit, then launch the full D-v02-2 pool-size study (§8.1 measure) with the new model, detached
  (`setsid nohup`), time limit 3 h (Ansh lifts the 60-min cap for this run). **If either fails:** no launch; report.

### 8.3 Addendum (2 Nov 2026, before the new code): fix (d) — the min rule inside the classifier's split region (Ansh's decision, option (i))

**8.2 validation result** (commit `ab24977`; seeds 100–109, 30 runs per model): EventGPC **P1 13/30 — FAIL**, P2 27/30 — pass;
P3 medians (old → EventGPC): weak 94.5 → 34.0, medium 102.5 → 49.5, strong 102.5 → 65.5 picks.

**Reading:** most wrong-event T points are near-ties: the true glide–split gap has a median of 0.025 GPa, and 34 of 54 have a τ_P
error below 0.02 GPa. A few are confident and costly. For example, in medium seed 103 the model gives p = 0.87–0.96 at
χ = +30 and τ_P errors up to +0.168 GPa. This happens because the classifier alone picks the event, and the two GP surfaces
are never compared, although the measured τ_P is by definition the lower of the two.

**Decision (Ansh, 2 Nov 2026):** option (i): keep P1 and P2 exactly as written; replace EventGPC by fix (d).

**New model fix (d), class `EventGPD`** (`EventGP` and `EventGPC` kept unchanged for the record):
- Same classifier, same two regression GPs, same settings as §8.2.
- **Prediction:** if p = P(split | x) ≤ 0.5: event glide, τ_P = glide GP mean. If p > 0.5: τ_P = min(glide GP mean, split GP mean),
  and the event is whichever is lower.
- **AL spread:** standard deviation over 256 samples. The event is drawn from Bernoulli(p). In the glide branch, τ comes from the
  glide GP; in the split branch, τ = min(glide sample, split sample).
- 0 split points: as §8.2 (glide). 0 glide points: as §8.2 (split).

**Pass criteria (unchanged from §8.2)**, now on FRESH seeds 110–119 (100–109 have been seen), all three surfaces, random
selection on the 2000-point pool, run protocol as §8.2 (≥ 100 picks), EventGPC and EventGPD on the same seeds and random
orders (60 runs):
- **P1.** At the final pick, wrong-event predictions on T ≤ 1 point, in ≥ 27 of 30 runs (EventGPD).
- **P2.** After pick 30, no RMSE rise > 0.05 GPa between consecutive 10-pick checks, in ≥ 27 of 30 runs (EventGPD).
- **P3.** Median picks needed per surface, EventGPC vs EventGPD (descriptive).
- **Descriptive only, NOT a pass condition:** for each model, the number of wrong-event T points at the final pick with
  |τ_P error| > 0.02 GPa, and the largest |τ_P error| at a wrong-event point.
- **Tests:** the 43 existing ones pass, plus (1) the two-patch case for EventGPD (no split predicted between the patches);
  (2) p > 0.5 with the glide GP mean lower → EventGPD predicts glide with the glide mean.
- **If P1 and P2 pass:** `scripts/v02_pool_study.py` → `MODEL = EventGPD` (no other change), commit, launch the full D-v02-2
  study detached (`setsid nohup`), time limit 3 h. **If either fails:** no launch; report where the wrong-event points are.

### 8.4 POST-HOC criterion change (made after seeing results) — 2 Nov 2026, committed before the third validation run

- **History:**
  - §8.2, EventGPC: P1 13/30 — FAIL (seeds 100–109, `ab24977`).
  - §8.3, EventGPD: P1 12/30 — FAIL, P2 29/30 — pass (seeds 110–119, `786a685`).
  - This is the third attempt.
- **Reason:** P1 as written counts every wrong label, including near-ties where the true glide–split gap is below the τ
  resolution. T near-tie counts (true gap below 0.02 GPa; `data/synthetic_v02/T_near_ties.txt`): weak 3, medium 5,
  strong 9 points. The v0.2 claim is about τ_P accuracy (0.02 GPa target), and a wrong label matters only when it gives a
  wrong τ_P. The 0.02 GPa threshold below is the pre-registered RMSE target, not chosen from these results.
- **Model fixed:** EventGPD (§8.3), no changes.
- **New pass criteria:** FRESH seeds 120–129 (100–119 have been seen and are NOT rescored), all three surfaces, random
  selection on the 2000-point pool, run protocol as §8.2 (≥ 100 picks), EventGPD only (30 runs):
  - **P1′.** At the final pick, at most 1 T point with BOTH a wrong event AND |τ_P error| > 0.02 GPa, in ≥ 27 of 30 runs.
  - **P2.** Unchanged: after pick 30, no RMSE rise > 0.05 GPa between consecutive 10-pick checks, in ≥ 27 of 30 runs.
  - **P3.** Median picks needed per surface (descriptive).
  - The original P1 count (≤ 1 wrong-event point, any error) on the same runs is reported as **descriptive only**.
- **Stop rule:** this is the last validation attempt tonight. If P1′ or P2 fails, no further model or criterion changes
  tonight.
- **Overnight rule (fixed now, before the result):** the pool-size study is launched tonight in both cases.
  - If P1′ and P2 pass, it is the study of record.
  - If either fails, it is launched as PROVISIONAL: the output folder name ends in `_provisional`, and the log header and
    NOTES say "PROVISIONAL: model EventGPD not validated; result usable only if EventGPD is later accepted unchanged under a
    dated addendum; otherwise rerun". A provisional result is never counted as a validation.
- **README commitment:** the v0.2 README will report the original P1 failures, this post-hoc change and its reason.

### 8.5 Extension decided after seeing the 3N/5N/8N result (2 Nov 2026, Ansh)

- **Result** (pool-size study of record, output `49bb451`; 609 runs, clean): N = weak 33, medium 45, strong 47. §8.1 measure
  (median random − AL picks, 90 % bootstrap interval):
  - weak: 3N −10.0 [−15.0, −2.0], 5N +11.0 [+3.5, +17.0], 8N −19.0 [−35.0, +4.0] → chosen 5N (164 points);
  - medium: 3N −7.5 [−15.5, −2.0], 5N −3.5 [−12.5, +1.0], 8N +0.5 [−19.0, +7.0] → chosen 3N (136 points);
  - strong: 3N +0.5 [−4.0, +3.5], 5N +6.0 [−1.0, +15.0], 8N +20.0 [+4.0, +30.0] → still growing at 8N.
  The §8.1 rule therefore gives no size.
- **Decision (Ansh):** extend the **strong** surface only, to **12N and 16N**. Everything else as before:
  - same seeds 0–19, same methods (AL, random, space-filling, grid), same model EventGPD;
  - same §8.1 measure and D-v02-2 edge rules (a median below the interval = stopped growing; exactly on the upper end =
    growing).
- **Weak (5N) and medium (3N) choices stand** and are not rerun. Overall size = the largest chosen size across surfaces;
  nights = ceil(size / 50).
- **Cap, fixed now, before the result:** if strong is still growing at 16N, the pool = 16N (752 points), and the README states
  that the advantage was still growing at the cap.

### 8.6 Pool size of record (3 Nov 2026)

By the §8.1 rule and the §8.5 extension (outputs `49bb451`, `be8bc0e`):
- weak 5N = 164 points; medium 3N = 136 points;
- strong 8N = 376 points: the 12N median +3.0 is below the 8N interval [+4.0, +30.0], so the advantage stopped growing.

**Overall candidate pool P = 376 points; nights = ceil(376 / 50) = 8.**

Descriptive, not a pass condition: on the synthetic surfaces, AL beat the minimum of the other methods in 0–7 of 20 seeds in
every cell, and space-filling had the lowest median picks in almost every cell.

### 8.7 Gate A-yz calibration ladder and ramp top (3 Nov 2026, Ansh; decided after the 2 Nov smoke test, before any gate data)

- **Finding:** the smoke test at χ = 0, σ_yz = 0.25 (`d294da1`) failed its model check at τ = 0.5 on the loading angle:
  0.377° against the 0.1° limit. v0.1 at the same χ gave 0.034°.
- **Cause:**
  - The v0.2 strain model is a global quartic (p = 0–4, 5 coefficients, least squares) fitted to 7 points. Below 1.0 GPa its
    only point is τ = 0.
  - In that gap it departs from the calibrated ends: at τ = 0.5 the model gives γ_xz 6.96 × 10⁻⁵, against 11.21 × 10⁻⁵ on the
    straight line between τ = 0 and 1.0.
  - Shear modulus ~70 GPa × 4.2 × 10⁻⁵ ≈ 0.003 GPa, which matches the measured σ_xz error of 0.0033 GPa (inference).
  - v0.1 was safe because its correction had no constant or linear term (zero value and slope at τ = 0), with 3 coefficients
    on 4 points.
- **Decision:**
  - Whenever σ_yz ≠ 0: calibrated τ = (0, 0.5, 1.0, 1.5, 2.0, 2.4, 2.7, 3.0); model checks τ = (0.25, 0.75, 1.25, 1.75, 2.2,
    2.55, 2.85).
  - Unchanged: the model form and the Gate A criterion (D2, with the τ = 0 floor reading), and the σ_yz = 0 behaviour.
- **Ramp:**
  - `tools/peierls_chi.py --coarse` becomes required (no default), so a missing ramp fails instead of silently topping out at
    2.4 GPa. v0.2 runs pass `0:3.0:0.05` (§8 Q5).
  - No past run relied on the old default: every script passed `--coarse` explicitly.
- **Cost:** about +3 s per calibration (one more calibrated τ).

### 8.8 Gate A tolerance floor (3 Nov 2026, Ansh; decided after smoke tests S1/S2, before any gate data)

- **Finding** (`fc0cb2f`): S2 (χ +30, σ_yz 1.0) PASS 15/15. S1 (χ 0, σ_yz 0.25) FAIL 14/15, failing only the τ = 0.25 model check
  (σ_xz error +0.00104 GPa, angle 0.24°).
  - The strain model's absolute σ_xz error is about 0.001 GPa at every τ. S1: +0.00104 at 0.25, −0.00104 at 1.25, −0.00099
    at 2.55.
  - The angle test allows 0.001745 × τ, so any check below ~0.6 GPa fails by construction.
  - This is the same structure as the P1 change of §8.4: a relative tolerance that cannot be met as its denominator goes to
    zero.
- **Decision:** in the v0.2 Gate A check (`tools/gate_a.py gate_a3_check`, which `calibrate_yz.py` uses), every off-axis
  tolerance that is currently 0.001745 × max(τ, |σ_yz|) becomes **0.001745 × max(τ, |σ_yz|, 1.0 GPa)**.
  - At τ ≥ 1.0 GPa nothing changes. Below it the tolerance is held at 0.0017 GPa (1.7 MPa, one sixth of the 10 MPa τ
    resolution).
  - Unchanged: the 1 % branch, the other-components 2 % rule and the ladders. v0.1 code and results are untouched.
  - Implementation note (Claude): the σ_xz angle branch at τ > 0 was written as |Δσ_xz| ≤ 0.001745 × τ. It now uses the same
    max(τ, |σ_yz|, 1.0) as the other off-axis tolerances. Over the ladder (|σ_yz| ≤ 1.0 GPa) this equals 0.001745 × τ at
    τ ≥ 1.0, so nothing changes there either.
- **Reason:** the loading angle matters where the dislocation can move (τ_P; lowest v0.1 value 1.165 GPa). Below that, 1–2 MPa
  of off-axis stress is far below the τ resolution.
- **Risk to report later:** if any v0.2 τ_P is below 1.0 GPa, its loading angle may exceed 0.1° (up to 0.1° × 1.0 / τ_P). The
  comparison stage must flag such points.
- **README commitment:** report this change together with the §8.4 change.
- **Phase rule (Ansh, 3 Nov 2026; wording corrected before the first gate run):** a criterion is frozen before the data it
  judges is seen.
  - Each gate's criteria freeze before that gate's first run.
  - All comparison criteria (claim rule, RMSE target, P1′, P2, pool size, step-3 verdict) freeze before night T.
  - Changes before that point need an independent reason in a dated addendum. Any change after the judged data has been seen
    is post-hoc by definition, is reported as such, and needs Ansh's explicit decision.
  - (The first wording, "all criteria frozen from the first gate run", would have frozen steps 4–6 before their definitions
    were checked.)

### 8.9 Step 3 (Gate A-yz) definitions (3 Nov 2026, Ansh; calibration phase, before the first gate run)

- **Run A is descriptive only** (reported, never a gate). It answers "does σ_yz disturb the v0.1 strain model?".
  - Model: the χ = −30, 0, +30 entries of `checks/07_chi_calibration/strain_model.json`. The driver first verifies that
    `checks/08_gateA_rebox/strain_model_al.json` holds identical copies of these three and reports any difference.
  - σ_yz by a plain surface force, no correction, applied first at τ = 0, then τ (§8 Q1).
  - τ = 0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.4 GPa: the v0.1 validated range only, no extrapolation above 2.4. Each is a
    single measurement without iteration.
  - Per-τ verdict by `gate_a3_check` (§8.8). Per-pair summary: all pass or not, largest angle error, largest other_pct.
- **Run B** (`calibrate_yz.py`, ladders of §8.7, rule of §8.8) **runs for EVERY pair**, regardless of Run A, one call per pair.
  - Reason: every T and P point carries its own calibration, so the real loading never uses the v0.1 model at σ_yz ≠ 0.
    Run A is a physics diagnostic.
  - This replaces the plan's earlier "Run B only if A fails".
- **σ_yz = 0 pairs:** Run A (as a v0.1 reproduction check) and Run B with the σ_yz = 0 ladder (re-calibration to the 3.0 GPa
  top).
- **Step-3 verdict:**
  - s_A = the largest s in {0.25, 0.5, 0.75, 1.0} such that Run B passes GATE_A3 (all its τ rows) for every pair with
    0 < |σ_yz| ≤ s, all three χ, both signs.
  - **Step 3 PASS** iff the three σ_yz = 0 pairs pass Run B AND s_A exists (≥ 0.25).
  - Otherwise **step 3 FAIL**: v0.2 stops for Ansh's decision (post-hoc under the phase rule).
  - s_A feeds the §8 Q2 range rule unchanged: it is one input to s, not the final s.

### 8.10 Step 4 (core-must-not-move) definitions (3 Nov 2026, Ansh; before step 4's first run)

- **Runs:** χ −30, 0, +30 × sign (+, −) = 6 runs on the dislocation cell. Per run:
  - σ_yz 0 → ±0.75 GPa (s_A) in 0.25 steps at τ = 0;
  - then τ 0 → 0.6 GPa in 0.05 steps at the top σ_yz;
  - then 2 stability seeds at the top state (existing stability-test settings, unchanged);
  - then unload in reverse: τ 0.6 → 0 in 0.05, then σ_yz → 0 in 0.25.
- **Loading for τ > 0 at each σ_yz level:** the step-3 Run B entry for that (χ, σ_yz) pair in
  `checks/10_gateA_yz/strain_model_yz.json`. No level above s_A.
- **Checks at every state:**
  - the core centre (existing core locator) is within 0.1 Å of its start position, with no jump to another site;
  - the stability seeds return;
  - after unload, atom positions are within 0.01 Å and the energy within 10⁻⁴ eV of the initial relaxed state;
  - the largest per-atom non-affine displacement is recorded at every state, descriptive only.
- **Verdict:**
  - s_4 = the largest σ_yz level such that every state up to and including it passes in all 6 runs. The τ steps and the
    stability seeds count at the top level only; a run failing at τ > 0 fails its top level.
  - s = min(s_A, s_4).
  - **Step 4 PASS** iff s_4 ≥ 0.25; otherwise FAIL, and v0.2 stops for Ansh's decision (post-hoc).
- **D-v02-3 (Ansh, 3 Nov 2026; recorded before any step-4 data):**
  - (a) The per-atom non-affine maximum is **descriptive only**, with no pass/fail threshold. σ_yz is meant to change the core;
    failure is caught by the same-site / no-split check, the stability seeds and unload reversibility.
  - (b) **Pre-registered review flag:** if the per-atom non-affine maximum in any state of a run exceeds **0.5 Å**, the run is
    marked **REVIEW** and that state's atom snapshot is kept. 0.5 Å is about b/5 (b = 2.47 Å); an easy-to-hard core change
    moves core atoms about b/3 = 0.82 Å. The flag does **not** change s_4.
- **D-v02-4 (Ansh, 3 Nov 2026): the six choices left open in §8.10 are frozen as implemented in `run_step4.py`, with:**
  1. The step-3 entry is also used at τ = 0. The measured off-axis stress at τ = 0 is logged against the Gate A floor
     (descriptive).
  2. Core check = centroid within 0.1 Å of the start state AND top column < 1.4 b. The failing condition is recorded
     separately ("site jump" / "split").
  3. The v0.1 minimiser settings (CG, etol 0, ftol 1e-6, dmax 0.01 Å). The number of "linesearch alpha is zero" stops and the
     final force norm per state are reported (descriptive).
  4. Reversibility reference = the start state. The start state vs `relaxed_down.data` difference is reported (descriptive).
  5. Stability seeds 1 and 2 with the script defaults. Kick seeds are a separate namespace from surrogate/AL seeds (the
     130+ rule does not apply).
  6. The `.npy` copies are git-ignored, as implemented.
- **D-v02-5 (Ansh, 3 Nov 2026; recorded before any step-4 data):**
  - (a) **Non-affine reference clarified.** The per-atom non-affine displacement for the REVIEW flag (D-v02-3 b) is measured
    from the **START state** of the run (cumulative), with the affine part removed by the same least-squares fit. The
    step-to-step value is kept as a descriptive column. Reason: a gradual core change of ~0.82 Å over 15 loading steps is
    ~0.05 Å per step, so a step-to-step reference would never trip the 0.5 Å flag.
  - (b) **Fallback rule.** The final s_4 must have passed the full sequence (ladder, τ ramp 0 → 0.6, 2 stability seeds,
    unload) in all 6 runs. If the highest passing level is below 0.75, step 4 is **INCOMPLETE**, and a follow-up run of the full
    sequence at that level is required before s is used. Reason: otherwise a fallback level would be granted without facing
    shear.

### 8.11 Steps 5 and 6 criteria — D-v02-6 (Ansh, 3 Nov 2026; before any step-5 or step-6 data)

§5.4 rules 3–4, step-table row 6 (L78) and L163 stay as written, with:
- **(a) σ_yz = 0 first.** Step 5 first runs σ_yz = 0 at χ −30, 0, +30 with the v0.2 driver. These are τ_P(0) for rule 4.
  - Reason: the v0.2 loading uses the step-3 recalibration, so v0.1 values could fake or hide a 0.02 GPa effect.
  - If any midpoint differs from v0.1 by > 0.02 GPa, stop before the corners. References: χ −30 1.160–1.170, χ 0
    1.310–1.320 (B2), χ +30 2.040–2.050 GPa.
- **(b) Corner failure.** ALL six corners repeat at s − 0.25, plus the D-v02-5 (b) step-4 full-sequence follow-up at that
  level. Until both are done, step 5 is INCOMPLETE; a failure at 0.25 → stop for Ansh.
  - Reason: the probes are T nodes at ±s, so all of them move with s.
  - This replaces "the failed probes are repeated" in rule 3 (a pre-data change, reason on record).
- **(c) Step 6 pass rule.** Each 2× run is within 3 % of its 1× value (the v0.1 rule). The absolute difference is reported
  against 0.02 GPa (descriptive). A failure → stop for Ansh.
  - The calibration carries to the 2× cell as an **assumption**: the calibrated quantities are intensive, and the forces
    come from the cell's own area.
  - The size test cannot separate a loading error from a size effect; to be disclosed in the README.
- **(d) Chaining:** step 6 starts only if step 5 PASSes at 0.75 and rule 4 is not triggered.
- **D-v02-7 (Ansh, 3 Nov 2026; before any step-5 or step-6 data):** the seven NOT STATED choices are frozen as implemented in
  `ramp_v02.py`, `run_step5.py` and `run_step6.py` (commit `33f7b92`), except:
  - **Item 1:** Phase A also requires rule 3 (first event a D5 glide jump or easy → split, no anomaly; both seeds return).
    A failure → STOP (exit 2). Reason: Phase A reproduces v0.1, which passed rule 3.
  - **Item 5:** step 6 requires, per 2× run, the same first-event D5 class as its 1× run and both seeds returning; otherwise
    STOP for Ansh. Reason: comparing τ_P across different mechanisms is not a size test.
  - **Item 6:** the step-4 follow-up below 0.75 is built only if needed.

### 8.12 Night T and night P — D-v02-8 (Ansh, 4 Nov 2026; before any T or P data)

Steps 3–6 passed (s = 0.75; NOTES S17). These decisions fix how T and P are computed, sealed and used. They supersede the
§4.2 regular-part wording where stated.

1. **Sealing.** All model, AL and comparison settings freeze before night T. After that, people may read T values. Only the
   comparison script reads T results (`data/sealed/T_results.csv`), and it is run once.
2. **T nodes already computed.** The 3 phase-A runs (χ −30, 0, +30 at σ_yz 0) and the 6 step-5 corners (the same χ at
   σ_yz ±0.75) count as T nodes: 9 of the 49 are done. They are imported with their provenance (`checks/12_corner_gate`, commit `b1bec26`).
3. **P = `make_pool(376)`** (`src/surrogate/designs.py`), exactly as in the synthetic study: 45 χ cell centres
   (−29.33 … +29.33°, step 1.33°) × 8 σ levels (±0.125, ±0.375, ±0.625, ±0.875 × s), plus χ = ±30 at the same 8 levels.
   At s = 0.75 the levels are ±0.09375, ±0.28125, ±0.46875, ±0.65625 GPa. This supersedes the §4.2 regular-part wording.
   Reason: the pool size of record was validated on this layout. P is disjoint from T (no P σ level is a T level).
4. **Order of computation of P:** a fixed random permutation of the 376 points, `numpy.random.default_rng(20261104).permutation(376)`.
   **Seed 20261104** (value chosen by Claude, recorded here before any P data; not one of the surrogate seeds below).
5. **Comparison seeds (§8 Q8, 20 seeds) = 130–149.** "Fresh seeds" rule: any new validation uses seeds never used before.
   Used so far: 0–19 (pool-size study), 100–129 (validations §8.2–§8.4), 130–149 (the comparison).
6. **Model and AL settings: carried over unchanged from the synthetic study** (§8.1, §8.2, §8.3; `src/surrogate/event_gp.py`
   `EventGPD`, `scripts/v02_pool_study.py`):
   - inputs x = (χ / 30, σ_yz / s), with s = 0.75 GPa; target y = τ_P bracket midpoint (GPa); label = D5 event of the first
     move (glide jump → glide, easy → split → split);
   - regression GPs (glide GP on glide points, split GP on split points): Constant(1, [1e-3, 1e3]) × RBF([0.5, 0.5],
     [1e-2, 1e2]) + White, noise std learned in [0.0029, 0.1] GPa (D4); targets divided by max(std of all τ_P, 0.05 GPa);
     glide prior mean = mean of glide points; split prior mean = highest observed τ_P; a GP with < 3 points keeps its
     initial hyperparameters; 2 optimiser restarts; `random_state` = seed;
   - event classifier: `GaussianProcessClassifier`, Constant(1, [1e-2, 1e2]) × RBF([0.5, 0.5], [1e-2, 1e2]), Laplace,
     2 restarts, `max_iter_predict` 100, on the labels of all picked points; 0 split points → p = 0; 0 glide points → p = 1;
   - prediction (EventGPD): p ≤ 0.5 → glide, glide mean; p > 0.5 → min(glide mean, split mean), event = the lower one;
   - AL: seeded random start of 3 points (the first 3 of `random_order(|P|, seed)`), then the point of largest spread =
     std over 256 draws (event ~ Bernoulli(p); glide branch: glide sample; split branch: min(glide sample, split sample));
     draw generator `default_rng(seed + 1000)`; refit after every pick;
   - random: `random_order(|P|, seed)`; space-filling: greedy maximin in x, first point drawn with `default_rng(seed)`;
     grid: nested regular sub-grids of the regular part from 2 × 2 up (`grid_designs`), deterministic, one value;
   - picks needed = the first pick with RMSE ≤ 0.02 GPa on T; a method that never reaches it within P counts as |P| + 1 (Q9).
7. **P1′ / P2 on real data:** descriptive only, computed on T.
8. **Invalid points** (Gate A fail, anomaly, no bracket, failed stability seed): recorded as invalid with the reason, excluded
   for all methods equally, and the count is reported. **Stop for Ansh if > 2 of the 49 T points or > 18 (5 %) of the 376
   P points are invalid.**
9. **§8.8 flag:** the comparison lists the points with τ_P < 1.0 GPa and reports RMSE with and without them (descriptive).
10. **Calibration:** every new (χ, σ_yz) pair gets its own `calibrate_yz.py` calibration and Gate A verdict (step-3 Run B
    criteria, §8.8 floor) before its ramp. There is no interpolation. Reasons: the v0.1 rule (every new χ is calibrated); the
    quadratic-form hypothesis is untested; the cost is ~2.7 h for the 404 new pairs (28 for T, 376 for P). **Parked:** a fit
    of the quadratic form to all calibrations, descriptive, after the data.

**Implementation notes (Claude, 4 Nov, written with the driver, before any T or P data; for Ansh to confirm before night T):**
- (a) **Pre-ramp levels** (§8 Q1: σ_yz first, at τ = 0, in steps ≤ 0.25 GPa). A T point uses T's own σ levels of the same
  sign up to its target (0.25, 0.5, 0.75: identical to steps 4–5). A P point uses P's own σ levels of the same sign up to its
  target (0.09375, 0.28125, 0.46875, 0.65625 GPa: steps of 0.094, then 0.1875). Every level is therefore a pair of the same set
  with its own calibration, as in steps 4–5 (D-v02-4 item 1). A point whose target or level pair fails Gate A is invalid
  ("Gate A fail").
- (b) **Start state at a χ with no σ_yz = 0 entry** (P χ values other than 0, ±20, ±30): zero correction. At τ = 0, σ_yz = 0
  every σ_yz = 0 entry (q2–q4) gives exactly zero, so this is the same loading, not an approximation.
- (c) **Acceptance mapping of item 8** (from `ramp_v02.summarise`): no bracket = no first move up to 3.0 GPa, or a move during
  the pre-ramp; anomaly = the first D5 event on the path is an anomaly; failed seed = a stability seed did not return. The
  step-5 "below the ramp top minus one coarse step" check is recorded but does not make a point invalid (a bracket at
  2.95–3.00 GPa is still a bracket).
- (d) **Production strain-model file** `checks/14_production/strain_model_prod.json`: the 26 step-3 entries copied unchanged
  plus the new production entries. `checks/10_gateA_yz/strain_model_yz.json` (the step-3 record) is not modified.
- (e) **Night batches:** night N = the first 50 P points not yet done in the permutation order, fixed in
  `data/production/P_night<N>.json` at the first launch of that night, so a resumed night reruns the same batch.

**D-v02-9 (Ansh, 4 Nov 2026, before any new T or P data):**
1. **Implementation notes (a)–(e) above are confirmed as implemented** (pre-ramp levels of the point's own set, zero
   correction at a χ with no σ_yz = 0 entry, acceptance mapping, production strain-model file, fixed night batches).
2. **Grid contender with invalid points:** an invalid grid node is replaced by the nearest valid pool point in
   (χ / 30, σ_yz / s) distance; ties go to the lower pool index; the number of replaced nodes is reported.
3. **Interim T rule:** no one reads T values until `scripts/v02_compare.py` is committed and tested on synthetic data
   (its commit hash recorded here). Until then only counts are shown (points done / valid / invalid, with the invalid
   reasons); `run_production.py --set T` prints no τ_P, bracket or event values. The 9 step-5 T values already seen
   (step-5 gate, `checks/12_corner_gate`) are disclosed in the README.
4. **Nightly limit 600 min** (`timeout 600m`); a night that overruns resumes with the same command.
5. **NP = 7 MPI ranks, fixed for all production data** (T and P).

**D-v02-10 (Ansh, 4 Nov 2026; recommended by the planning desk at 09:55 during night T, confirmed by Ansh after night T, before any
T value was read by the comparison script):**
Disclosure, not a criterion change. During night T the following were seen: (1) cumulative elapsed times for T01, T02, T04, T05
on progress lines (Ansh, planning desk); (2) ramp state counts for the same 4 points in Claude Code tool output, later quoted to
the planning desk (Claude Code, Ansh, planning desk); (3) T07's (χ −20, σ_yz −0.75) state rows, including its bracket and
first-event class, shown by the Claude Code "files changed" panel while the driver wrote `data/sealed/T_states.csv` (Ansh,
planning desk) — with the 9 step-5 values this makes 10 of 49 T values seen; (4) run-wide time-left estimates (negligible). A
status check at 13:30 was confirmed by Ansh to show no sealed rows. All model, active-learning, pool and comparison settings were
frozen before night T (8.8) and no method uses T while picking, so no result can be affected. The driver was not changed mid-run
(one dataset must come from one code version). Fixes for future blind sets: print only run-wide elapsed time; send no prompts
while a blind set is being written, or write sealed output outside the project folder.

*Log-file note (same commit):* the calibrator rewrote committed `checks/10_gateA_yz/log.yz_ref` at start-up (CPU-time lines only;
no physics change). It is restored from HEAD, and from night P1 on the production calibrator writes its log under
`checks/14_production/` instead. This changes an output path only; P1 has not started.

**D-v02-11 (Ansh, 4 Nov 2026, after night T, before any T value was read; comparison-script details):**
A fact check (4 Nov, 16:49) found the verdict rules already frozen: Q8 "AL wins seed k only if AL_k < min(random_k,
space-filling_k, grid)" (a tie is not a win); 8.12(6) "picks needed = the first pick with RMSE ≤ 0.02 GPa on T" (first
crossing); 8.12(6) space-filling first point drawn with `default_rng(seed)`. No criterion is changed. Added:
1. Descriptive: wins / ties / losses reported as counts and percentages of the 20 seeds; each tie labelled by type (both
   reached at the same pick count, or neither reached, both |P| + 1); a tie is never called a loss.
2. Descriptive: for each method and seed, whether RMSE stayed ≤ 0.02 GPa for all later picks, and the largest RMSE after the
   first crossing.
3. **Seal guard:** `scripts/v02_compare.py` reads `data/sealed/T_results.csv` only with `--final`, and refuses (exit message,
   no T values printed) unless all hold: no modified or staged tracked files (`git status --porcelain --untracked-files=no` is
   empty); pytest 0 failed; the script's commit hash is recorded in V02_PLAN; sha256 of `T_results.csv` equals
   `5bec6137421be78a3222783df8a54490b1b72631cdcb8b63ab434887148813bc`; no final output file exists yet (the comparison is run
   once, 8.12(1)). Without `--final` it runs on synthetic data only. Tests: a modified tracked file, an existing final output,
   and a wrong sha256 each make `--final` refuse.
4. **Synthetic test before `--final`:** (a) hand-checkable unit tests with known answers: an invalid point excluded for all
   methods; a tie counted as a tie, not a win; never reached = |P| + 1 = 377; grid nearest-valid substitution with the
   lower-index tie-break; first crossing at pick 11 for RMSE 0.021 / 0.019 / 0.022 / 0.018 at picks 10–13; (b) on the
   pool-study synthetic surfaces, the script reproduces the per-seed results recorded in
   `data/synthetic_v02/pool_study_runs.json` exactly, for every configuration recorded there, by reusing the pool-study scoring
   code (`run_order`, `run_al`, `run_grid`; scoring from 3 picks; grid GP `random_state` 0); if any value differs, stop and
   report, do not adjust; (c) a rerun gives identical output. The synthetic verdict tests the code only and is not reported as
   a result.

**D-v02-12 (Ansh, 4 Nov 2026, before night P1):** a production night interrupted mid-point resumes that point from its last
saved state (existing driver design, as in steps 4–6), not from scratch. Every ramp state is a separate LAMMPS run started from
the previous state's data file, so a resumed point reads the same input an uninterrupted run would; 0 K results are
deterministic. Acceptance: the resumed point's state and result rows are identical, field by field, to an uninterrupted run; no
state that already has a saved row is rerun; no duplicate or orphan rows; a state's row is written only after its data file is
completely written. The planning desk's earlier "from scratch" check is superseded.

**D-v02-12a (Ansh, 4 Nov 2026, before night P1):** the resume test found that `n_new_calib` and `calib_seconds` counted
calibrations done since the current start, not for the point, so a resumed point under-reports them (bookkeeping only; physics
rows identical). Fix: both fields count the point's own rows in `calibrations.csv` (`for_point` column). The D-v02-12
acceptance is unchanged: every result and state field identical to an uninterrupted run, except clock-time fields. Night T never
restarted; its recorded values are checked to equal the new definition.

**D-v02-13 (Ansh, 4 Nov 2026, during night P1, before any T value was read; comparison-script readings, from the part-1 build of
`scripts/v02_compare.py`, `4f0744e`):**
- (a) P1′ and P2 (descriptive, 8.12 item 7) are evaluated on each run's model at its first crossing, or at its last pick if it
  never reaches the target.
- (b) Invalid P points (8.12 item 8): every method builds its order exactly as on the full P (the seeded random permutation,
  space-filling from the seeded first point, AL's seeded starts and candidate set), and an invalid point is skipped when it comes
  up; skipped points do not count as picks. This replaces the part-1 reading (orders built on the valid subset), so that the
  registered seeds give the registered order for every valid point.
- (c) A grid substitute (D-v02-9 item 2) that is already in the design is not added twice.
- (d) The D-v02-11 4b reproduction runs in first-crossing mode, which is what the pool study recorded (picks needed); the full Q9
  curves are descriptive and run only in `--final`. Part 2 adds `--workers` (parallel over seeds) with a test that parallel
  output equals sequential output exactly; the reproduction and `--final` run only when no LAMMPS job is running.
- (e) Every detached run echoes its exit code (the part-1 smoke run did not).

**D-v02-14 (Ansh, 5 Nov 2026, after the part-1 code review, before the replay):** (1) Replay mode --reproduce: reads data/synthetic_v02/pool_study_runs.json and replays every recorded run (surface, method, M, seed) through the same imported run_order/run_al/run_grid with the pool study's own settings (random cap N_CAP = 400), in first-crossing mode; every field each run records is compared exactly; any difference -> stop and report, nothing adjusted. Before running, it prints the run count, the fields compared, any recorded field first-crossing mode cannot produce, and an estimated time. pool_study_ext_runs.json is out of scope (D-v02-11 names only pool_study_runs.json). (2) The replay refuses to start unless OMP, MKL and OPENBLAS threads are all 1. (3) In first-crossing mode, "stayed <= 0.02 afterwards" and "largest RMSE after the first crossing" are stored as None (not observed), never True. (4) Seal hash: V02_PLAN gets one dedicated line "v02_compare.py commit: <full 40-character hash>" (added at the hash-record decision after the replay matches); the guard requires an exact full-hash match on that line only. (5) Synthetic output: verdict fields are placed under "synthetic_tally_code_test_not_a_result". (6) model_keys_all stays in scripts/v02_compare.py (moving it would touch a passing test for no gain). (7) The docstring is updated together with D-v02-13 (b). The replay is not the same as running the comparison: the study holds 39 configurations (609 runs, incl. random at M = 2000), so exact reproduction means replaying each recorded run.

**D-v02-15 (Ansh, 5 Nov 2026, after the part-2 build; no P point invalid so far, so no result depends on it):** AL under invalid points (D-v02-13 b for an adaptive method). The seeded 3-point start is taken from the full-P random order with invalid entries skipped. At each later pick AL predicts over the full-P candidate set (all not-yet-picked pool points, invalid ones included) with the same random generator, and takes the valid point with the largest spread; any invalid point ranking above it is removed from the candidates and not counted as a pick. The spread is a 256-draw estimate whose random numbers are drawn as one block over the candidate list (src/surrogate/event_gp.py:147-151; pick step scripts/v02_pool_study.py:83, scripts/v02_compare.py:248), so a candidate's spread depends on the list's length and order; this rule therefore does not in general pick the same point as ranking the valid points only. Once a point is invalid, no rule can reproduce the all-valid run, because AL might have picked it. The rule is chosen to be fixed before any real-data comparison, seeded and deterministic, and identical to ps.run_al when no point is invalid (tests/test_compare.py:73). Order disclosure: this reading was implemented in a0fe086 before it was accepted here; no real-data comparison had been run.

**D-v02-16 (Ansh, 5 Nov 2026, during night P2, before the replay and before any real-data comparison):** cost meter, descriptive only. (1) Picks stored: scripts/v02_compare.py saves, per run (method, seed), the ordered list of picked P IDs up to and including the first crossing (the last pick if never reached); for grid, the design that crossed (with any substitutes); invalid skipped points are not in the list. This change is committed before the replay, and the full script hash is recorded only after the replay matches with this version. (2) Cost of a method's run = the sum, over its picked points, of each point's measured ramp seconds, plus every calibration key that the picked points' states use (keys from P_states), each counted once per run, at that key's measured seconds (calibrations.csv). CPU-hours = seconds x 7 / 3600 (NP = 7 fixed). Times were measured under real load (night and daytime runs) and are labelled as such. (3) Two savings are reported: (a) each method's cost to its first crossing against the measured cost of all 376 P points ("simulate everything"); (b) AL against the best other method, paired per seed, in core-hours, as median and range over the 20 comparison seeds. Negative values are reported as negative. A method that never reaches the target is reported as "not reached, full P cost", with no saving. (4) Descriptive only: the Q8 verdict stays the frozen rule on pick counts. If picks and core-hours disagree, both are shown and the verdict does not change. (5) No new method: v0.2 has no separate bisection baseline (bisection is inside every point); only the registered methods plus the "simulate everything" reference are costed. (6) Money: rupees = core-hours x price per core-hour; the price is an argument, sourced at README time with date and link; core-hours are always shown next to rupees. Any DFT figure in the README is one line labelled "estimate" (picks saved x a cited per-point DFT cost); a measured DFT saving is v0.3 work. (7) The meter is a separate script, scripts/v02_cost.py, reading --final's output plus P_results.csv, P_states.csv and calibrations.csv; it is built with hand-checkable tests on fake data, committed, and its commit hash recorded in V02_PLAN before --final, so the cost rules are fixed before any result is seen.

**D-v02-17 (Ansh, 5 Nov 2026):** the replay of data/synthetic_v02/pool_study_runs.json matched exactly (609/609 runs, every recorded field except wall-clock seconds) with scripts/v02_compare.py at the commit below; replay log logs/replay_v02.out, 39,560 ConvergenceWarnings (the 5,051 quoted for the pool study is the count in the seed-0 timing run, logs/v02_pool_timing.log; logs/v02_pool_study.log holds none, so there is no like-for-like count), wall time 693 s (11 min 33 s, 7 workers). This is the script --final must run.
v02_compare.py commit: 48ab3f9a3ae5b992834393084947234e036c7548

D-v02-18 (6 Nov 2026): cost meter rules for scripts/v02_cost.py (extends D-v02-16; descriptive only; the Q8 verdict is unchanged). Accepted by Ansh line by line.
 1. Point seconds: a picked point's cost is the sum of its P_states.csv "seconds" over all its states (all phases). On P1-P3 this equals P_results.ramp_seconds within 0.6 s, and it stays correct after a stop/resume.
 2. Calibrations: each distinct non-zero model_key in the picked points' states is counted once per run, with its "seconds" from calibrations.csv, also when the row was made for a T point (the run would have needed it). sigma_yz = 0 keys are excluded (exact zero correction, no calibration). If a key has more than one row, its PASS row is used.
 3. Cost window: picks up to and including the first crossing of RMSE <= 0.02 GPa on sealed T.
 4. Never reached: a run that never crosses is costed at all its picks (full budget), marked "not reached (>=)"; any saving against it is a lower bound.
 5. AL vs best other: best other = the single method among random, spacefill and grid with the lowest median cost over seeds 130-149, among runs that reached the target; paired with AL per seed; report median and range of the saving, negatives kept; name the method. Not the cheapest method per seed, because that choice cannot be known in advance.
 6. Grid substitutes: a substituted grid point is costed as the point actually simulated.
 7. Grid in pairing: grid is one deterministic run, so the same grid cost is paired with every seed.
 8. Starting design: every training point is counted, including the starting design; a build test checks picked count = starting points + added picks.
 9. Invalid points: none in P1-P3; if any appear in P4-P8, the summary states that cost may be understated by up to that many points, because the frozen v02_compare does not store skipped points.
 10. Gate A failures: a failed point adds only its calibration seconds, and only in the simulate-all baseline.
 11. Stop/resume: state seconds only; seconds lost to a killed in-flight state are reported separately as "lost to stop/resume" (from log times) and charged to no method.
 12. Day vs night: seconds are used as recorded, labelled "real load", not normalised (P1-P3 ramp means 595 / 610 / 577 s).
 13. Simulate-all baseline: all 376 points' state seconds, invalid points included, plus each distinct non-zero key once.
 14. Outputs: cost_runs.csv (method, seed, picks, reached, n_points, point_s, n_keys, calib_s, total_s, core_h) and cost_summary.json (baseline, savings (a) and (b), price and its date only if --price is given, and the sha256 of every input file: P_results.csv, P_states.csv, calibrations.csv, the --final output).
 15. AL selection compute: reported separately, descriptive, from timing recorded by --final; if --final does not record it, state "not included" and give an estimate labelled "estimate".
 16. core-h = seconds x 7 / 3600, always shown; rupee price sourced with date and link at README time.
v02_cost.py commit: 35816a81d9dc8494568d2eed60113eb80faf5c1e
D-v02-19 (6 Nov 2026): three cases D-v02-18 left open, filled by Claude Code in v02_cost.py at 35816a8 BEFORE this decision (disclosed; second time after D-v02-15). Accepted by Ansh line by line.
 1. If AL itself never reaches the target, AL is costed at its full budget (>=) and saving (b) is marked "upper bound (AL not reached)".
 2. A calibration key with no PASS row uses its last row (only possible for Gate A failures; none so far).
 3. The AL selection-compute estimate (median 8.7 s per run, 20 synthetic AL runs at pool 376, pool_study_runs.json) is converted as s / 3600 core-h (one thread per run), not x 7; labelled "estimate".
D-v02-20 (6 Nov 2026, before the code change): bounded savings in saving (b). Accepted by Ansh line by line.
 1. If neither AL nor the best other reaches the target on a seed, that seed's saving is "not defined": listed per seed with this label, no number computed.
 2. The median and range of saving (b) use only seeds where both AL and the best other reached. Always shown next to them: n seeds AL not reached (upper bound), n seeds other not reached (lower bound), n seeds neither reached (not defined). Every bounded seed stays in the per-seed list with its label. Reason: excluding either kind of bounded seed alone would bias the median for or against AL; the counts show what is left out.
v02_cost.py commit (D-v02-20, supersedes 35816a8): ce362527b5948d5167c73be67d137fb02169a4a5

D-v02-21 (8 Nov 2026, before --final): --final run rules. Accepted by Ansh line by line.
 1. Command, from the project root: tools/launch_safe.sh final_v02 bash -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate ml && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/v02_compare.py --final --workers 7; echo "EXIT_CODE=$?"' (7 workers as the replay; threads set explicitly because the code only uses setdefault and checks threads only in --reproduce; not niced, nothing else heavy runs).
 2. Before launch: P8 committed; git clean (PROMPTS.md and NOTES.md committed); pgrep shows 0 LAMMPS, mpirun, hydra_pmi_proxy and run_production processes (the code has no LAMMPS-running check). During the run: charger in, lid open, no other heavy work.
 3. Header pre-check: only the first line (column names) of data/sealed/T_results.csv is read and compared with the columns read_real uses; column names only are printed, never a value. This is the only change to the names/sizes/sha256 rule, and only for column names.
 4. Figure pre-check: matplotlib (Agg) saves a test PNG to the scratchpad in the ml env; data/v02_compare/ and figures/ can be created and written to.
 5. If final.json is written but final_curves.csv or the figure fails: final.json is the result, there is no rerun; the CSV and figure are rebuilt from final.json only, by a separate small script, disclosed.
 6. If --final refuses or crashes before final.json exists: nothing is consumed; the fix is decided first; any edit to scripts/v02_compare.py means a new replay and hash record (D-v02-17).
final.json sha256: 117fb650591a961f08664b0541f48a3873c8b849945627ac7bf06a1c0a54afec (--final run once, 8 Nov 2026, scripts/v02_compare.py at 48ab3f9)

D-v02-22 (9 Nov 2026, before the v0.2 publish): publish rules. Accepted by Ansh line by line.
 1. checks/14_production/calib/ logs (11,320 files, 71 MB) stay private; calibrations.csv and strain_model_prod.json (every verdict and model) are published; README says the raw logs are available on request.
 2. P_states.csv and the step 4-6 result CSVs are published unchanged, local path column included, so their sha256 match the values recorded in cost_summary.json and NOTES; README says the path column is a local record and not needed.
 3. data/sealed/T_results.csv and T_states.csv are published, since --final has run once; this lets anyone check final.json against T.
 4. NOTES.md and PROMPTS.md are not updated in v0.2 (the public copies stay as in v0.1; current versions hold personal details). V02_PLAN.md is published in full. A separate CATCHES.md (catches only, no personal details) may follow as a normal commit after the tag. CLAUDE.md is updated only if a scan finds no personal details or paths.
 5. scripts/v02_diag_kernels.py and checks/11_core_gate/resume_check_crash_18-11.py (hard-coded paths, one-off diagnostics) stay private.
 6. The public commit is built on branch public by checking out named paths from master (never merge or rebase); a fresh clone of public must pass pytest -q tests in the ml env, else stop and decide; the annotated tag v0.2 goes on the public commit only, after confirming its parent is 2ae9d55; pushes only with an explicit PUSH line.
