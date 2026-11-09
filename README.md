# BCC iron screw-dislocation Peierls stress: LAMMPS workflow and Gaussian-process surrogate (v0.2)

**Anshuman Choudhury** · PhD, computational materials science (Univ. Paris-Saclay / CEA Saclay) · co-author, *Unravelling the jerky glide of dislocations in body-centred cubic crystals*, [Nature Materials 23, 47–51 (2024)](https://www.nature.com/articles/s41563-023-01728-5) · [LinkedIn](https://www.linkedin.com/in/anshuman-choudhury-513kite/)

**v0.1** (28 Oct 2026) computed the Peierls stress against the orientation of the applied shear (χ) and tested a small surrogate model that chooses the next simulation, with each prediction written down before the simulation runs. **v0.2** (9 Nov 2026, this release) adds a second stress (σ_yz), runs 425 loading simulations, and tests, pre-registered and on a sealed test set, whether letting the surrogate choose the simulations saves any.

## At a glance

- **What I built:** an automated LAMMPS workflow that computes the Peierls stress of the ½⟨111⟩ screw dislocation in bcc iron (the lattice resistance the dislocation must overcome to move) as the loading orientation (χ) and an additional shear stress (σ_yz) vary, and a Gaussian-process surrogate that predicts this stress, with uncertainty, across that range.
- **Scale:** 425 stepped loading runs, all valid: a pool P of 376 points, which I ran as 8 unattended production runs over 4–8 Nov 2026 on 7 MPI ranks, and a sealed test set T of 49 points (40 from a separate run on 4 Nov, 9 carried over from an earlier step). Every P input was checked against its own calibration entry (0 mismatches); the pool alone took 458.9 core-hours.
- **What it achieved:** the surrogate reached the 0.02 GPa accuracy target on the sealed test set with every point-selection method tried. The space-filling and grid designs needed a median of 109 and 115 of the 376 pool points, and the grid run used 70.4 % less compute than simulating the whole pool.
- **What I tested:** whether uncertainty-based active learning reaches that target with fewer simulations than random, space-filling or grid selection. I fixed the test and its pass rule in advance and ran it once on the sealed test set. Active learning did not win (1 of 20 seeds; 15 were needed), and I report the result as found.
- **Why it can be trusted:** my rule for this project was that correctness has no compromise. Every decision is dated in V02_PLAN.md; a seal guard allowed the final comparison to run only once; the comparison script replayed 609 recorded runs exactly before that; negative controls showed the input checks catch a 0.1 % change; 133 tests. Problems the checks caught are listed under [What was caught along the way](#what-was-caught-along-the-way-v02).
- **Skills:** dislocation physics and atomistic simulation (LAMMPS, EAM potential, stress-loading calibration); Gaussian-process surrogates with uncertainty; design of experiments and active learning; pre-registered evaluation on held-out data; planning and running long unattended compute campaigns with safe stop and resume; reproducible Python (pytest, pinned environment); leading AI coding agents with independent verification.

## v0.2 result (pre-registered, sealed test, run once)

**Question.** Does uncertainty-based active learning (AL) need fewer LAMMPS runs than random choice, a space-filling design
or a regular grid to reach RMSE ≤ 0.02 GPa on a sealed, held-out set T of Peierls stresses over (χ, σ_yz)? Every method
picks from the same pool P of 376 points; T holds 49 points (7 × 7: χ −30° to +30° in 10° steps, σ_yz 0, ±s/3, ±2s/3, ±s
with s = 0.75 GPa), and no method sees T while picking.

**Data.** P: 376 of 376 points valid, computed in 8 production runs (P1–P8, 4–8 Nov 2026). T: 49 points, 0 invalid. Same potential, cell and
temperature as v0.1: Mendelev et al. 2003 potential #2, the 14.84 × 149.4 × 146.87 Å cell (27,972 atoms), 0 K.

**Result** (`data/v02_compare/final.json`; 20 seeds per method, grid is one deterministic run):

| Method | Reached target | Median picks | Range | Stayed ≤ 0.02 GPa after first crossing |
|---|---|---|---|---|
| AL | 20 / 20 | 217.5 | 97–278 | 6 / 20 |
| random | 20 / 20 | 175.0 | 56–331 | 7 / 20 |
| space-filling | 20 / 20 | 109.0 | 36–209 | 1 / 20 |
| grid | 1 / 1 | 115 | (one run) | 0 / 1 |

**Verdict.** `q8`: `{"al_wins": 1, "needed": 15, "of": 20, "saving_claimed": false}` (tally: 1 win, seed 131; 19 losses;
0 ties). The pre-registered claim rule needed 15 wins of 20, so no saving is claimed. On this problem, space-filling and
grid designs reached the target with the fewest simulations; uncertainty-based AL did not beat them.

**How the test was kept fair.** The question, the accuracy target and the claim rule were fixed in V02_PLAN.md before the comparison was run, and every later decision is dated there (D-v02-1 to D-v02-22). T was sealed: `--final` refuses to run unless git is clean, the script's commit matches the hash recorded in V02_PLAN.md (48ab3f9), T's sha256 matches its recorded value, the tests pass and no earlier result exists, so the comparison on T ran once. Before that run, the script replayed all 609 recorded synthetic runs exactly (D-v02-17). Every LAMMPS input of the 376 P points was checked against its own calibration entry (0 mismatches), and two negative controls that change the entries by 0.1 % were confirmed to be caught.

**Cost** (`data/production/cost/`; descriptive only, the verdict is the rule on pick counts; seconds as recorded under real load; core-h = seconds × 7 / 3600; no price). Simulating all of P: 236,015.9 s (458.9 core-h). Grid, the cheapest other method by median cost, reached the target at 69,792.3 s, saving 70.4 % of the all-of-P cost. AL's median run cost 134,314.55 s (`cost_runs.csv`), saving 43.1 %. Paired per seed against grid, AL cost more in 19 of 20 seeds (median −64,522.25 s). AL's own selection compute is not included: `--final` records no per-run timing; for scale only, an estimate from 20 synthetic AL runs at pool size 376 is 8.7 s per run (median).

![RMSE on T versus picks](figures/final_rmse_curves.png)

*RMSE on T against the number of points picked from P, per method: median over seeds, band = range over seeds; dashed line =
the 0.02 GPa target.*

**Limits.**
- One interatomic potential (Mendelev 2003 #2), 0 K, a straight dislocation.
- Cell size: at χ 0, σ_yz 0, the 2× cell (293.75 Å along glide) gives 1.295 GPa against 1.315 GPa in the 1× cell, a
  0.020 GPa difference that equals the 0.02 GPa target. Absolute accuracy is therefore limited by cell size; all methods use
  the same data, so the comparison between them is unaffected. The size test cannot separate a loading error from a size
  effect.
- The loading calibration is carried to the 2× cell as an assumption (the calibrated quantities are intensive, and the
  forces come from the cell's own area).
- In the 2× cell, 88 of 88 unmoved relaxations and 5 of 6 stability relaxes stopped at "linesearch alpha is zero" (fnorm
  ≤ 6.9e−4 eV/Å; the 1× cell: 8 of 279). Production uses the 1× cell.
- Values of T seen before the seal: see the Disclosure under "Pre-registered v0.2 test" below, and D-v02-10 in
  `V02_PLAN.md`.
- Post-hoc changes, both disclosed: (1) the pre-set model check P1 counted wrong event labels at near-ties finer than the
  data resolve; two model fixes both failed it, and it was replaced post hoc by P1′ (count only wrong labels that also give a
  τ_P error above 0.02 GPa), then tested on fresh seeds 120–129: P1′ 30/30, P2 30/30, original P1 17/30 (§8.4). (2) Two sets
  of rules, the AL rule under invalid points (D-v02-15) and three cost-meter cases (D-v02-19), were implemented in code
  before they were accepted.
- The Gate A off-axis tolerance has a floor (§8.8): below τ = 1.0 GPa it is held at 0.0017 GPa, so a point with τ_P below
  1.0 GPa may have a loading angle above 0.1°. 7 of the 49 T points have τ_P below 1.0 GPa (`low_tau_nodes` in final.json).
- The effect of σ_yz depends on its sign: the σ_yz −0.75 GPa runs show about 2× the cumulative non-affine displacement of the
  +0.75 GPa runs.
- σ_yz here is the stress on the fixed (−110) plane; how it maps onto the non-glide stress of Gröger, Bailey & Vitek (2008)
  has not been checked.
- AL's spread is a 256-draw Monte Carlo estimate, so AL's picks carry sampling noise; this is part of the registered method
  (D-v02-15).
- The AL selection-compute figure in the cost line is an estimate from synthetic runs, not a measurement on `--final`.
- Not published: the raw calibration logs in `checks/14_production/calib/` (11,320 files, 71 MB); every verdict and fitted
  model is in `calibrations.csv` and `strain_model_prod.json`, and the raw logs are available on request.
  `data/production/P_states.csv` and the step 4–6 result CSVs are published unchanged, so their sha256 match the recorded
  values; their path column is a local record and is not needed.

**How to reproduce.** `conda env create -f environment.yml`, `conda activate projecta-ml`, then `pytest -q tests`.
With `OMP_NUM_THREADS=1`: `python scripts/v02_compare.py --synthetic medium --seeds 130` runs the comparison on synthetic
data, and `python scripts/v02_compare.py --reproduce` replays the recorded pool study
(`data/synthetic_v02/pool_study_runs.json`). `--final` refuses to run in a clone: `data/v02_compare/final.json` already
exists, and the comparison on T is run once only. `data/sealed/T_results.csv` is published, so final.json can be checked
against T. Every decision, with its date, is in
[`V02_PLAN.md`](V02_PLAN.md).

## What was caught along the way (v0.2)

Under my rule that correctness has no compromise, every result had to pass independent checks. These are some of the problems the checks caught, each before it could affect the published result.

1. **A check that tested only half of itself.** The input checker compares every LAMMPS input with its calibration entry. Its negative control (all entries changed by 0.1 %) turned out to reach only the per-point checks, not the calibration section, so that part had never been tested. A second negative control was added, and from then on both had to flag every entry on every production run.
2. **A stop that had never really been tested.** The stop-and-resume test made a fake simulation fail inside the program; it never stopped a real process. The first real stop showed that ending the main process left its child processes running; the second showed that stopping the simulation before its driver lets the driver see a failure. No data was affected: the stop procedure now stops the driver first and confirms nothing is left, and the two real stops lost 21 s of computing in total.
3. **A reader that had never seen real data.** The code that reads the sealed test set had never run on real files, and the final result file is written before the figures, so a late crash could have used up the single run. Before the run, the sealed file's column names (names only, no values) were checked, the figure step was tested, and a rule was fixed for rebuilding the figures from the result file if needed.
4. **A fair opponent for active learning.** On cost, active learning is compared with the single method that is best on median over all seeds, not with whichever method happened to be cheapest on each seed, which would need hindsight. Runs that never reach the target give only bounds, so they are kept out of the median and counted separately.
5. **A pre-registered check that failed.** The model check P1, set before the data, counted wrong event labels even at near-ties finer than the data can resolve, and two model fixes both failed it. It was replaced openly, after the fact, by P1′ and tested on fresh seeds (120–129); the change and the original failure are both reported under Limits.
6. **A claim in a design note that was false.** A draft rule stated that one way of handling invalid points was equivalent to ranking only the valid points. A check before acceptance showed it was not (the active-learning spread is a 256-draw estimate), and the rule was accepted without that claim (D-v02-15).
7. **A cost figure of the wrong kind.** The only available number for the active-learning model's own computing time comes from synthetic runs, not from the final run. It is shown for scale only, with the reason: even a hundred times larger it would be under 1 % of active learning's simulation cost, and leaving it out can only flatter active learning.
8. **Private files in the history (v0.1).** Before the first publication, potential files that may not be redistributed and private logs were found in the project's history, so removing them from the current files would not have been enough. The public repository was built as a separate, clean branch.
9. **This page's own first draft.** Two sentences in the first draft of this page claimed more than the record supports: that all 425 runs came from the 8 production runs, and that every decision was accepted line by line. The fact check required before publishing caught both, and the wording was corrected to match the record.

## Question (v0.1)
How does the 0 K Peierls stress of the ½⟨111⟩ screw dislocation in bcc iron depend on the orientation of the applied shear
(the MRSS angle χ), and can a small surrogate model with an honest uncertainty choose the next atomistic simulation, with
its prediction written down before the simulation runs?

## Method (v0.1)
- **Atomistic:** LAMMPS (22 Jul 2025 update 6), Mendelev et al. 2003 EAM potential 2 (`potentials/Fe_mm.eam.fs`, fetched by
  `tools/fetch_potential.sh`, see Potential), 0 K
  energy minimisation (CG, force-converged: ftol 10⁻⁶ eV/Å, max step 0.01 Å).
- **Cell:** one straight screw dislocation in a slab, periodic along the line (x = [111]) and the glide direction
  (z = [−1−12], periodic array of dislocations with a b/2 box tilt), free surfaces normal to y = [−110].
  14.84 × 149.4 × 146.87 Å, 27,972 atoms.
- **Loading at angle χ:** σ_xy = τ cos χ by forces on two surface slabs; σ_xz = τ sin χ by box strains calibrated on a
  perfect slab (with a second-order correction), set by rewriting the data file.
- **Peierls stress:** ramp τ from 0 in 50 MPa steps, refine in 10 MPa steps; τ_P is the first stress at which the compact
  easy core at its start site is lost: either a **glide jump** (the core moves at least one lattice step, 2.331 Å, to an
  equivalent compact site) or an **easy → split transition** (the core becomes centred on one atom column). Read from
  differential-displacement (DD) maps. Each "last unmoved" state is checked by a stability test (small random kick,
  re-minimise, must return to the same state).
- **Surrogate** (`src/surrogate/`, PyTorch and scikit-learn): Gaussian process (GP) with its noise level learned between the
  bracket noise (0.0029 GPa) and 0.1 GPa; a 5-member MLP ensemble (bootstrap + randomized priors) and a "Schmid-ratio" form
  were also tested. Calibration check: fraction of left-out values inside ±1 and ±2 predicted spreads.
- **Active learning:** the model picks the angle with the largest predicted spread; its prediction and ±2 band are written,
  time-stamped, to `data/al_live_predictions.csv` before the simulation starts.

## Potential
Mendelev, Han, Srolovitz, Ackland, Sun & Asta, *Phil. Mag.* 83, 3977–3994 (2003), potential #2 for Fe (eam/fs). The file is
not in this repository. `tools/fetch_potential.sh` downloads it from the NIST Interatomic Potentials Repository (entry
2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2, version 1) and checks its SHA-256; its tables are identical to the
`Fe_mm.eam.fs` used for every run (only the comment lines differ). `tools/fetch_potential.sh --lammps` fetches that
byte-identical file from the LAMMPS repository instead. NIST's version 3 (2009) of the same entry has different tables
and was not used.

## Quick start (no LAMMPS needed)
```
conda env create -f environment.yml          # pinned versions actually used (pip: requirements.txt)
conda activate projecta-ml
python -m pytest tests                       # 133 tests; in a fresh clone 132 pass and 1 is skipped (it needs per-atom energy dumps that are not published)
cd scripts
python fit_vitek.py                          # Vitek-type fit and comparison: data/vitek_fit.json, data/vitek_compare.csv
python plot_peierls_vs_chi.py                # figures/peierls_vs_chi.png   (from data/audit_moves.csv, data/vitek_fit.json)
python plot_surrogate.py                     # figures/surrogate_chi_v1.png (fits the surrogate to data/surrogate_train_v1.csv
                                             #   and data/al_live_results.csv)
```
Other analyses that run from `data/` alone: `scripts/surrogate_loo_d4.py` (leave-one-out), `scripts/al_retro_d4.py`
(retrospective active learning against random choice and bisection).

To re-run simulations: `conda env create -f environment-lammps.yml`, `tools/fetch_potential.sh`, then the run scripts in
`sims/` (each LAMMPS run is started through `tools/launch_safe.sh`). The large per-step data files and run logs are not in
the repository; the per-step results are in `data/*.csv`.

## Validation chain (v0.1; all computed here; references labelled)
1. Lattice constant 2.85532 Å and cohesive energy −4.12244 eV/atom vs 2.855 Å / −4.122 eV/atom (Mendelev 2003): 0.011 %.
2. C11 / C12 / C44 = 243.33 / 145.01 / 115.99 GPa vs 243.4 / 145.0 / 116.0 GPa (Mendelev 2003, as supplied): ≤ 0.03 %.
3. Potential file identical to NIST "Potential #2" version 1 (30,000 / 30,000 table values).
4. Screw core compact and non-degenerate (three b/3 arrows, symmetry deviations ≤ 0.001 b), as reported for this potential
   by Gilbert et al., PRL 111, 095502 (2013).
5. Cell size: twice the glide length changes τ_P by −0.8 % at χ = 0 and by 0.0 / +0.9 % at χ = +30 / −30 (target < 3 %).
6. Loading on a perfect slab, every angle used: calibrated points σ_xy, σ_xz within 0.1 %, other stresses ≤ 0.08 % of τ;
   interpolated stresses σ_xz within 0.9 % (or a loading tilt ≤ 0.04°), others ≤ 0.24 % of τ.
7. χ = 0: τ_P = 1.310–1.320 GPa vs 1.21 GPa (Chaussidon, Fivel & Rodney, *Acta Mater.* 54, 3407 (2006), same potential):
   **+8.7 %**.
8. pytest: 21 tests (one skips on a fresh clone; it needs per-atom energy files that are not in the repository) (lattice, elastic constants, b, atom counts, tracker, ML code, the "moved" rules, the loading criterion,
   the noise floor, the benchmark formula) pass.

## Verification gates and what they caught (v0.1)
Each item: what looked wrong, and what it was. Full detail, with dates and numbers, in `NOTES.md` ("Things I caught").
- **Reference values that were our own output.** The "published" a₀ and E_coh we first compared against matched our own run
  digit for digit; we switched to the paper's precision.
- **Which potential file.** The file header doesn't say "potential 2"; a table-by-table comparison with NIST settled it, and
  showed that NIST's 2009 revision differs.
- **Frame and cell choices.** A single dislocation periodic along glide needs a box tilt along the line, so the axes were
  relabelled. The periodic cell carries a built-in non-glide shear (about ±0.7 GPa); a cell twice as long tested it.
- **Hard core relaxes away.** A core started at the hard site slid to the easy site, so the easy core was found by trying
  both.
- **Memory.** The first builder needed > 10 GB for the large cell and crashed the laptop; it now builds plane by plane.
- **CPU count.** `nproc` reports 1 once OMP_NUM_THREADS = 1; and using all 8 cores was 6× slower than 7.
- **Energy rise misread.** A +0.84 eV step was the normal elastic rise per 10 MPa, not a new core.
- **The Burgers triangle is not the core shape.** The triangle holding b can sit beside the core; the column with the
  largest DD sum shows where the core is.
- **Lattice units.** LAMMPS's oriented lattice gave ⅔ of the true z repeat; caught by a zero-strain stress check.
- **A box tilt is not a pure shear.** In the [111] frame an xz tilt also creates normal stresses of 44 %; hence the
  calibrated multi-strain loading.
- **Second-order stresses.** The surface-force loading leaves ~3 % non-glide stress at τ_P; corrected for χ ≠ 0 and, from
  Session 13, for χ = 0 too.
- **A tracker that depends on its window.** The energy-centroid core position moved by up to 0.8 Å with window size; DD maps
  are used instead.
- **A bug that looked like physics.** A χ = 0 rerun gave 1.25 instead of 1.28 GPa. The first explanation ("minimiser path
  dependence") was wrong: LAMMPS's `change_box … remap` stretched the slab by 10⁻⁴ on every call. Fixed by rewriting the
  box ourselves; the result then agreed on 7 and 8 processes.
- **Refinement steps** added +10, +20, +30 MPa instead of +10 each time.
- **A jump the tracker missed** (1.98 Å read for a 2.331 Å move), so "moved" is now read from DD maps.
- **Garbage per-atom energies** for atoms on the cell boundary right after minimisation; fixed by re-evaluating first.
- **The ML ensemble was over-confident** (7 % inside ±1 spread); bootstrap + random priors fixed it on test data.
- **Live-AL recorder** took the coarse "moved" step instead of the fine one; the bad record was kept and the round it
  affected was voided.
- **A failed check did not stop a script** (`set -e` without `pipefail`).
- **A relative tolerance on a near-zero target.** "σ_xz within 1 %" failed at χ = 3° for a 0.037° tilt; the criterion now
  also accepts a loading-angle error ≤ 0.1° (changed after the failure, recorded as such).
- **The reference was a restatement.** ~1.25 GPa came from a later paper; the original gives 1.21 GPa.
- **A rule that depended on a label.** At χ = +28 the "moved" test first fired on a core change in place, then on a pure
  relabelling (atoms moved ≤ 0.013 Å). The current rule is label-free (easy → split or glide jump).
- **A stability test that ignored the loading model at χ = 0.**
- **A kink that was a protocol difference.** χ = 0 sat 0.03 GPa below its neighbours because it ran without the loading
  correction the other angles had; rerun with it, the point lies on the curve.

## Results (v0.1)
**χ curve** (146.87 Å cell; `data/surrogate_train_v1.csv`, `data/audit_moves.csv`, `figures/peierls_vs_chi.png`):

| χ (°) | side | τ_P bracket (GPa) | event | plane of a glide jump |
|---|---|---|---|---|
| −30 | twinning | 1.160–1.170 | easy → split | — |
| −29 | twinning | 1.160–1.170 | easy → split | — |
| −28 | twinning | 1.160–1.170 | easy → split | — |
| −24 | twinning | 1.180–1.190 | easy → split | — |
| −18 | twinning | 1.210–1.220 | glide jump | (−110) |
| −12 | twinning | 1.220–1.230 | glide jump | (−110) |
| −6 | twinning | 1.250–1.260 | glide jump | (−110) |
| 0 | — | 1.310–1.320 | glide jump | (−110) |
| +6 | anti-twinning | 1.390–1.400 | glide jump | (−110) |
| +12 | anti-twinning | 1.490–1.500 | glide jump | (−110) |
| +18 | anti-twinning | 1.640–1.650 | glide jump | (−110) |
| +24 | anti-twinning | 1.880–1.890 | glide jump | (−110) |
| +28 | anti-twinning | 2.110–2.120 | easy → split | — |
| +30 | anti-twinning | 2.040–2.050 | glide jump | (−1 0 1) |

- Twinning side easier, anti-twinning side harder: the ratio to Schmid's law (1.315 / cos χ) runs from 0.77 at −30° to
  1.35 at +30°.
- **The twinning-side points −24, −28, −30 (and +28) are easy → split transitions.** The split core is a known artifact of
  the Mendelev potential (Gilbert et al., PRL 111, 095502 (2013)), so these points characterise the potential, not iron.
- **Benchmark against Chaussidon et al. 2006 (same potential):** fitting τ(χ) = s0 / (cos χ + a cos(60° + χ)) to our
  (−110) glide jumps gives s0 = 1861 ± 18 MPa, **a = 0.82 ± 0.03**, against their s0 = 1604 MPa, **a = 0.61** (values as
  read by the planning chat). Same direction and shape; our asymmetry is stronger, and our values are 1–15 % above their
  fit. Why is open (Limitations).

![Peierls stress vs MRSS angle χ: glide jumps and easy → split transitions, Schmid curve, Vitek-type fits](figures/peierls_vs_chi.png)

**Surrogate, leave-one-out** (GP with learned noise, `data/surrogate_loo_d4_s14_summary.json`):
- all 14 points: RMSE 0.072 GPa, fraction inside ±1 / ±2 spreads 0.71 / 0.79 (ideal 0.68 / 0.95), learned noise 0.036 GPa.
  The large errors are all at +24, +28, +30, where an easy → split value (+28, 2.115 GPa) sits above two glide jumps
  (+24, +30); elsewhere the errors are ≤ 0.024 GPa.
- the 8 (−110) glide jumps only: RMSE 0.034 GPa, inside ±2: 0.88, learned noise at its floor (0.0029 GPa): the glide-jump
  curve is smooth once χ = 0 uses the same loading recipe as the other angles.

![Surrogate (GP with learned noise) with its ±2 band, the measured points and the three pre-registered predictions](figures/surrogate_chi_v1.png)

**Live active learning, pre-registered** (`data/al_live_predictions.csv`, `data/al_live_results.csv`):

| Round | χ | model | predicted (±1 spread) | ±2 band | measured | z | inside ±2? | event |
|---|---|---|---|---|---|---|---|---|
| 1 | −28° | GP, Schmid-ratio form, fixed noise | 1.158 ± 0.007 | 1.145–1.171 | 1.160–1.170 | +1.0 | yes | easy → split |
| 2 | +28° | same | 2.009 ± 0.009 | 1.990–2.027 | 2.110–2.120 | +11.3 | **no** | easy → split |
| 3 | −29° | GP, learned noise | 1.167 ± 0.045 | 1.076–1.257 | 1.160–1.170 | −0.04 | yes | easy → split |

- Round 2 missed because the model had never seen a split transition on the anti-twinning side: the +28 value lies
  0.07 GPa above the smooth curve through +24 and +30 (and above +30 itself), and the model's noise was fixed at the
  bracket noise, so its band was only ±0.02 GPa. After that, the noise level was made learnable.
- A third round picked in Session 12 was voided (picked from wrongly recorded data); its row stays in the file.
- Round 3 was a hit, but a weak test: after round 2 the learned noise rose to 0.038 GPa, so the band was ±0.09 GPa wide,
  and the pick lay between two measured neighbours (−30 and −28) with the same value.
- Largest predicted spread over the remaining candidate angles (same model): 0.022 GPa before any live round, 0.005 after
  round 1, 0.045 after round 2 (the +28 surprise), 0.043 after round 3 (−5 %). The loop did not make the model surer
  overall; the +28 point showed it a kind of event it cannot represent as a smooth curve.

**The honest active-learning finding:** on this one-dimensional curve, a fixed "measure the ends first, then bisect the
largest gap" rule matches or beats the spread rule: it is better at 5–6 points, and the spread rule is ahead at 8–9 points
by about 0.001–0.002 GPa, less than the learned noise. v0.1 shows a closed, pre-registered loop, not a saving in
simulations. v0.2, with two inputs (χ and a second stress component), is the real test.

## Pre-registered v0.2 test (written 28 Oct 2026, before any v0.2 data)
- **Inputs:** the MRSS angle χ in [−30°, +30°] and a second, non-glide stress σ_yz. The σ_yz range is set only after its
  own loading gate (a perfect-slab calibration like Gate A) passes.
- **Ground truth:** a dense grid of LAMMPS runs over (χ, σ_yz), held out from every method.
- **Metric:** the number of LAMMPS runs a method needs to reach RMSE ≤ 0.02 GPa on the held-out grid.
- **Contenders:** spread-based active learning, random choice, a regular grid, and a space-filling design; 20 seeds for each
  method with randomness.
- **Claim rule:** a saving is claimed only if active learning needs fewer runs than the best other method in at least 15 of
  20 seeds. Otherwise this README will report that active learning did not help.
- **Model:** the surrogate will know the event type (glide jump vs easy → split), since one smooth model cannot fit both.
- **Disclosure (4 Nov 2026):** 9 of the 49 held-out grid values were seen before the grid was sealed: the 3 phase-A runs
  (χ −30, 0, +30 at σ_yz 0) and the 6 step-5 corner runs (same χ at σ_yz ±0.75 GPa), computed as loading-gate checks. A
  further 1 of the 40 values (one grid point's bracket and event class) and some timing information were seen during the
  run through a tool display; see D-v02-10. All settings were fixed before the run, and no method uses these values when
  choosing points.

## Limitations (v0.1)
- The twinning-side values are easy → split transitions, a known artifact of this potential.
- Small cell (146.87 Å along glide). The cell-size test with the corrected χ = 0 loading is deferred to v0.2.
- Our asymmetry (a = 0.82) is stronger than Chaussidon's (0.61), unexplained. Candidate causes: our cell is periodic along
  the glide direction where theirs has free boundaries; the built-in non-glide shear of our periodic cell (but doubling the
  cell changed ±30 by only 0.0 / +0.9 %); and we compare our fit with their fit, not with their raw data.
- One smooth model for two kinds of event: across all points the learned noise is 0.036 GPa (12× the bracket noise) and
  LOO ±2 coverage is 0.79 (ideal 0.95). On the glide jumps alone the curve is smooth (noise at the floor, coverage 0.88).
  A model that knows which event to expect is not built.
- One interatomic potential, 0 K, a straight dislocation (no temperature, no kink pairs).

## How it was built
**v0.2.** I led this project as its scientist. I chose the interatomic potential, fixed in advance how the answer would be judged, and made or accepted every decision in V02_PLAN.md, the later ones line by line; each is dated there. I worked with two AI tools the way a lead works with a small team: Claude for planning and review, and Claude Code for writing and running the code. I did not take their work on trust: every step had to pass tests, exact replays, input checks with negative controls, and the single sealed run described above.

AI tools (Claude Code) wrote the code and ran the simulations. Anshuman Choudhury designed the verification gates, the tests
and every physics choice (potential, cell, loading, the "moved" definition, tolerances), and checked the key numbers.
Every result traces to a file in this repository: the per-step results in `data/`, the validation logs in `checks/`, and
the session-by-session record of what was run, why and what came out in `NOTES.md` (prompts in `PROMPTS.md`).

## Licence
MIT (`LICENSE`). The potential file is not covered: cite Mendelev et al. (2003) and the NIST repository.
