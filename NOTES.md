# Project A — Setup and Work Notes

Running log of what was done, why, and what came out. Plain language first, technical detail second.
Owner of the physics choices: Ansh. Code and commands written with AI tools; key numbers checked by Ansh.

---

## Everyday commands (the only ones worth remembering)

```
wsl                      # enter Ubuntu from Windows PowerShell
cd ~/projectA            # go to the project folder
conda activate lammps    # switch on the project's software environment
lmp -in <file>           # run a LAMMPS simulation from an input file
claude                   # start Claude Code inside ~/projectA
```

---

## Design choices (why we did it this way)

- **Validate before trusting:** every potential must reproduce its published basic values (lattice constant, cohesive energy, elastic constants) before any result is built on it. Example: Mendelev 2003 matched (2.8553 Å, −4.1224 eV/atom). The Proville file did not: it gives 2.8148 Å, against the 2.855 Å written in its own file header, so work on it was paused. The paper's own value is still to be checked.
- **What a basic LAMMPS input does:** build a BCC iron crystal; use periodic boundaries (an endless crystal, no surfaces); choose the interatomic potential; then minimise the energy at 0 K, with the box free to relax to zero pressure.
- **Why a conda environment:** it fixes the software versions and keeps them separate from everything else, so anyone can reproduce the work exactly.
- **Why WSL on the laptop, not Colab:** long simulations disconnect on Colab.
- **Elastic constants by small strains (Session 3):** stretch or shear the relaxed crystal by a tiny amount in both directions (+e and −e), read the change in stress, and divide by the strain. Using both directions cancels the first non-linear error. Two strain sizes (0.1 % and 0.5 %) are run, and agreement between them shows we are in the linear (Hooke's law) range. Shear strain is the *engineering* shear γ = 2ε_xy.
- **Every check gets its own folder and a named log file** (`checks/01_lattice/log.lattice`, …), so the logs are kept in git as evidence. The default `log.lammps` is not kept.
- **Single dislocation, periodic along the glide direction (Session 4):** PAD construction (Osetsky & Bacon 2003). Use the field of an infinite row of dislocations plus a box tilt of b/2 along the line, so both sides of the box fit exactly. The dislocation line lies along LAMMPS x because LAMMPS can only tilt along x.
- **Find the easy core by trying both triangles (Session 4):** build the core at an up- and a down-pointing triangle of atom columns, relax both, and keep the one that stays put with b/3 on all three central arrows.
- **Long runs go through `tools/run_with_progress.py`**, which writes iterations done / max, elapsed time and an upper-bound time left to `logs/<run>.log`.
- **Peierls stress by surface forces (Ansh's choice, Session 5):** apply the glide shear σ_xy as equal and opposite forces along the line (x) on a 2-atomic-plane slab at each free surface. Force per atom = σ·Lx·Lz / N_slab. Use `fix addforce` with `fix_modify energy yes`, so the minimiser's energy includes the work of the applied forces. Ramp in 50 MPa steps from the previous relaxed state, then refine in 10 MPa steps. The Peierls stress is the first stress at which the core moves at least one jump (a√6/3 = 2.331 Å) and keeps going.
- **Core tracking (Session 5):** excess-energy-weighted centroid (bulk −4.1224 eV/atom as zero, threshold 0.01 eV) of atoms within 4 Å of the glide plane, as a periodic mean along z.
- **Crash-safe launch for every LAMMPS run (Ansh's rule, Session 6):** start runs with `tools/launch_safe.sh <run> <command>`. It detaches the run from the terminal (`setsid nohup`), so a closed window can't kill it. It sends all screen output to `logs/<run>.out`, uses 7 of the 8 cores (`OMP_NUM_THREADS=1`, `mpirun -np $NP`, NP = cores − 1) and writes the PID to `logs/<run>.pid`.
- **χ ≠ 0 loading, option 2 (Ansh, Sessions 9–11):** σ_xy = τ cos χ by surface forces; σ_xz = τ sin χ by box strains (xx, zz, xz) taken from a perfect-slab calibration with a τ²–τ⁴ correction, so the extra normal stresses cancel (`checks/07_chi_calibration/`).
- **Set the box by rewriting the data file, not `change_box` (Session 11):** `tools/rebox.py` maps atoms affinely in x and z and leaves y untouched. `change_box … remap` stretched y on every call with the shrink-wrapped y boundary.
- **Force-converged steps (Ansh, Session 11):** CG with etol 0, ftol 1e-6 eV/Å (2-norm), `min_modify dmax 0.01` Å, so a step does not stop on a tiny energy change while the core is still creeping, and line searches cannot jump over a small barrier. The 2× cell reaches only fmax ~10⁻⁶–10⁻⁵ eV/Å (numerical floor).
- **Stability test for every "last unmoved" state (Session 11):** random ±0.001 Å kick, re-minimise at the same stress, check that the DD Burgers triangle is still on the same site. A state that passes is a true resting state, not a state the minimiser happened to stop in.
- **"Moved" = the compact easy core at the start is lost (D5, Session 14): a glide jump or an easy → split transition, read label-free from the DD map. Earlier: "the DD Burgers-triangle site changed" (Session 11; confirmed by Ansh 28 Oct; replaced by D1, then D5):** the direct test of Ansh's definition. The energy-centroid tracker only sets the 6 Å window and the halt guards; it can under-read a jump (1.98 Å for a 2.331 Å jump at χ = −30).
- **Per-atom dumps after `run 0` (Session 11):** re-evaluate before writing per-atom energies after a minimisation, so the values belong to the right atoms.
- **Surrogate learns the deviation from Schmid's law (Session 12):** the physics-informed models fit r(χ) = τ_P cos χ / τ_P(0) and convert back. Schmid's 1/cos χ is then built in, and leave-one-out RMSE drops by 37–50 %.
- **Pre-registered predictions in live active learning (Session 12):** the model's prediction and ±2 band for the picked angle are written, time-stamped, to `data/al_live_predictions.csv` before the run starts, and never rewritten. Only then can "the new value fell inside the band" be a real test.
- **Every new χ gets its own Gate A calibration (Session 12):** the strain model was calibrated only at the 6° angles. A new angle is calibrated with the same method and criteria (`checks/08_gateA_rebox/calibrate_chi.py`, through `rebox.py`) and must pass before its ramp runs.
- **Label-free state comparison (Session 14):** two relaxed states are "the same" if both are compact at the same Burgers-triangle site, or both are column-centred on the same column. Used by the "moved" rule and the stability test, so that a near-tie in the triangle label can't create or hide an event.
- **Every point of a curve from the same recipe (Session 14, D6):** χ = 0 now uses the same corrected loading and ramp as every other angle.
- **How the work is split:** code is written with AI tools (Claude Code). The physics choices are Ansh's, and Ansh checks the key numbers.

---

## Session 1 — Tue 20 Oct 2026: setup and first validation

### 1. Checked what was already on the laptop
- **What:** looked for WSL (Ubuntu inside Windows), Python and conda.
- **Found:** Ubuntu 24.04 already installed under WSL 2 (from an earlier setup); Windows Python 3.13; no conda; no LAMMPS; 954 GB free in Ubuntu.
- **Why it matters:** LAMMPS and scientific tools run most smoothly on Linux; WSL gives a real Linux system inside Windows without a second computer.

### 2. Installed Miniforge (conda) inside Ubuntu
- **What:** Miniforge, a version of the conda package manager pre-set to use conda-forge, the community channel that hosts LAMMPS.
- **Why Miniforge, not Miniconda:** same tool, but defaults to conda-forge and avoids Anaconda's licence prompts.
- **Result:** conda 26.7.2, installed at `~/miniforge3`.

```
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
~/miniforge3/bin/conda init bash
exec bash
```

### 3. Created a separate environment and installed LAMMPS
- **What:** an isolated "box" called `lammps` containing LAMMPS, Python 3.11, NumPy and Matplotlib.
- **Why an environment:** keeps this project's software versions fixed and separate from everything else, so the work can be reproduced exactly by anyone.
- **Result:** LAMMPS 22 Jul 2025, Update 6 (stable release). No compiling needed.
- **Useful find:** this build already includes the EAM styles (`eam`, `eam/alloy`, `eam/fs`) and machine-learning potential packages (ML-PACE, ML-SNAP, ML-IAP) — relevant to the solo-paper idea on ML potentials.

```
conda create -n lammps -c conda-forge lammps python=3.11 numpy matplotlib -y
conda activate lammps
```

### 4. Downloaded the first iron potential
- **What:** `Fe_mm.eam.fs` from the official LAMMPS repository, stored in `~/projectA/potentials/`.
- **Source (from the file header):** Mendelev, Han, Srolovitz, Ackland, Sun and Asta, *Phil. Mag. A* 83, 3977–3994 (2003).
- **Why this one first:** a standard, widely used iron potential with well-known published values — ideal for checking that the installation gives correct physics.

```
mkdir -p ~/projectA/potentials && cd ~/projectA
wget -P potentials https://raw.githubusercontent.com/lammps/lammps/stable/potentials/Fe_mm.eam.fs
```

### 5. Validation: lattice constant and cohesive energy of BCC iron at 0 K
- **What the simulation does:** builds a 5×5×5 BCC iron crystal (250 atoms) with periodic boundaries, then lets the box size relax to zero pressure while minimising energy (conjugate gradient). The relaxed box gives the lattice constant; energy per atom gives the cohesive energy.
- **Why:** before trusting any result, a potential must reproduce its own published basic properties. If these are off, everything downstream is wrong.
- **Input file:** `in.fe_lattice`

| Quantity | LAMMPS result | Published (Mendelev 2003) | Match |
|---|---|---|---|
| Lattice constant a₀ | 2.8553 Å | ≈ 2.855 Å | Yes |
| Cohesive energy E_coh | −4.1224 eV/atom | ≈ −4.122 eV/atom | Yes |

**Conclusion:** installation works and gives correct physics. Validation step (1) of Project A v0.1 started; elastic constants still to do.

---

## Session 2 — Tue 20 Oct 2026: Proville–Rodney–Marinica potential, first check (paused)

### 1. Set up Claude Code in the project
- **Install:** installed Claude Code inside Ubuntu (WSL) with the official native installer.
- **Result:** version 2.1.304, installed at `~/.local/bin/claude`.

```
curl -fsSL https://claude.ai/install.sh | bash
```

- **Fixed "claude: command not found":** the install folder wasn't on the PATH, the list of folders the shell searches for programs. Permanent fix:

```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

- **Started it:** ran `claude` inside `~/projectA` and logged in with the Claude Pro subscription through the browser. Auto mode is on: Claude Code approves low-risk actions itself and blocks risky ones.
- **What:** Claude Code now runs inside Ubuntu. Copied these notes into `~/projectA/` and wrote `CLAUDE.md`, the project rules Claude follows: always use the `lammps` environment, ask Ansh before any physics choice, update these notes every session, and never report an unchecked number.

### 2. Downloaded the Proville–Rodney–Marinica (2012) potential
- **What:** `MCM2011_eam.fs` (Finnis–Sinclair/EAM format) from the NIST Interatomic Potentials Repository, saved in `potentials/`.
- **Source:** NIST entry `2012--Proville-L-Rodney-D-Marinica-M-C--Fe`, implementation `2012--Proville-L--Fe--LAMMPS--ipr1`. Paper: Proville, Rodney and Marinica, *Nature Materials* 11, 845–849 (2012), doi:10.1038/nmat3401.
- **Note:** NIST says the file "was sent by M.-C. Marinica (CEA, France) on 10 January 2017 and posted with his permission". It's publicly posted on NIST, so it fits the "public potentials only" rule.
- **File check:** SHA-256 `3771064a6f50349ee03a4eccfdbaba26f1415d5c15c17820a6cded3d17943b0c`. Cutoff 5.5 Å. The header line lists a lattice constant of 2.85528 Å.

```
wget -P potentials https://www.ctcms.nist.gov/potentials/Download/2012--Proville-L-Rodney-D-Marinica-M-C--Fe/1/MCM2011_eam.fs
```

### 3. Lattice constant and cohesive energy at 0 K: surprise result
- **What:** the same test as Session 1 (250-atom crystal, box relaxed to zero pressure). Only the potential file was changed. Input `in.fe_lattice_prm`, log `log.fe_lattice_prm`.
- **Result:**

| Quantity | Our LAMMPS result | NIST's own calculation, same file | Header of the file |
|---|---|---|---|
| Lattice constant a₀ | 2.8148 Å | 2.8148 Å | 2.85528 Å |
| Cohesive energy E_coh | −4.1225 eV/atom | −4.12 eV/atom | — |

- **In plain words:** the crystal this file produces has atoms about 1.4 % closer together than its own header claims. Our run is correct: it agrees with NIST's calculation using the same file.
- **Extra cross-check:** we computed the energy of the unrelaxed crystal at five atom spacings (nearest-neighbour distance 2.40, 2.44, 2.48, 2.60 and 2.80 Å). All five match NIST's published energy-vs-distance table (`EvsR.Fe.txt`) to 8 or more digits. Input `checks/in.evsr_prm`, logs `checks/log.evsr_prm_*`.
- **Where the NIST numbers come from:** a NIST paper on vacancies in bcc screw-dislocation cores (tsapps.nist.gov, pub_id=921688). Its table gives this potential a₀ = 2.8148 Å and E_coh = −4.12 eV. The text calls the small lattice constant "the only unexplained outlier". These are NIST's own calculations with the file, not values from the original paper.
- **Not yet known:** the lattice constant the Proville 2012 paper itself reports. The paper is paywalled and wasn't checked.

**Decision (Ansh):** pause all work on this potential until the paper's own value has been checked.

---

## Session 3 — Tue 27 Oct 2026: re-check of the lattice and first elastic constants (Mendelev 2003)

### Setup
- **What:** made `~/projectA` a local git repository (no remote, nothing pushed). `.gitignore` skips `log.lammps`, `*.dump`, `__pycache__/`.
- **Software:** LAMMPS 22 Jul 2025, Update 6 (`lmp -h`), conda env `lammps`.
- **Potential:** `<project-root>/potentials/Fe_mm.eam.fs`, SHA-256 `d7b15f435a3f5192e8ed264d0dcbc052f1d5641ba5db610eab88199d77d20f7c`, unchanged. Header: "DATE: 2007-06-11 … CITATION: Mendelev, Han, Srolovitz, Ackland, Sun and Asta, Phil Mag A, 83, 3977-3994 (2003)".

### Check 1 — lattice constant and cohesive energy (re-run)
- **What:** same test as Session 1 (250-atom crystal, box relaxed to zero pressure with `fix box/relax iso 0.0`, CG minimiser, tolerances 1e-12/1e-12), moved into its own folder. It also saves the relaxed crystal for the elastic test.
- **Files:** `checks/01_lattice/in.lattice`, log `checks/01_lattice/log.lattice`, output `checks/01_lattice/relaxed.data`.
- **Command:** `cd checks/01_lattice && lmp -in in.lattice -log log.lattice`

| Quantity | Computed (this run) | Reference value* | Difference | Pass (≤1 %) |
|---|---|---|---|---|
| a₀ | 2.85532 Å (2.85532485998659) | 2.855 Å | +0.00032 Å (+0.011 %) | Pass |
| E_coh | −4.12244 eV/atom (−4.12243510206099) | −4.122 eV/atom | −0.00044 eV/atom (0.011 %) | Pass |

Residual pressure after relaxation: 0.0015 bar. Identical to the Session 1 run.

*Correction (27 Oct, later):* the a₀ and E_coh references are now the paper's precision, 2.855 Å and −4.122 eV/atom (Mendelev et al. 2003). The 4-decimal values first used here (2.8553 Å, −4.1224 eV/atom) were **our own 20 Oct output**, not published values.

### Check 2 — elastic constants C11, C12, C44 at 0 K
- **Plain words:** the elastic constants measure how stiff the crystal is. C11 is how hard it is to stretch along a cube edge, C12 is how much the sideways stress changes when you do that, and C44 is the resistance to shear. We stretch or shear the relaxed crystal by a tiny amount, both ways, and read off the stress.
- **Method:** start from `relaxed.data`. Each state is set with `change_box` (`x scale 1±e` for C11/C12, `xy final ±e·ly` on a triclinic box for C44). Atoms are then minimised at fixed box. C11 = −ΔPxx/(2e), C12 = −ΔPyy/(2e), C44 = −ΔPxy/(2e), with e for shear the **engineering** shear strain γ_xy = 2ε_xy. **Sign convention:** LAMMPS pressure P = −stress, in bar; 1 GPa = 10 000 bar.
- **Atoms don't move, as expected:** in a perfect bcc crystal under uniform strain every atom stays put by symmetry. The energy change from the minimisation was at most 2.1×10⁻¹¹ eV for the whole cell, and the largest leftover force was below 3×10⁻¹⁴ eV/Å. This is round-off: the minimisation changes nothing.
- **Files:** `checks/02_elastic/in.elastic`, `strain.mod` (one strained state), `derive.py` (B, C′, A); logs `log.elastic_e0.001`, `log.elastic_e0.005`.
- **Commands:** `cd checks/02_elastic && lmp -in in.elastic -var e 0.001 -log log.elastic_e0.001` (same with 0.005), then `python derive.py log.elastic_e0.001 log.elastic_e0.005`.

| e | C11 (GPa) | C12 (GPa) | C44 (GPa) | B (GPa) | C′ (GPa) | A |
|---|---|---|---|---|---|---|
| 0.001 | 243.3297 | 145.0091 | 115.9907 | 177.7826 | 49.1603 | 2.3594 |
| 0.005 | 243.3384 | 145.0053 | 116.0220 | 177.7830 | 49.1666 | 2.3598 |
| Reference* | 243.4 | 145.0 | 116.0 | — | — | — |

B = (C11+2C12)/3, C′ = (C11−C12)/2, A = 2C44/(C11−C12).

- **Linear range:** the two strain sizes agree to 0.009 (C11), 0.004 (C12) and 0.031 GPa (C44), far inside the ~1 GPa target.
- **Against the reference:** C11 −0.070 / −0.062 GPa (−0.03 %), C12 +0.009 / +0.005 GPa (+0.01 %), C44 −0.009 / +0.022 GPa (≤0.02 %). **All pass** (≤1 %).

\*Reference values: Mendelev et al., *Phil. Mag.* 83, 3977 (2003), potential 2, as supplied by Ansh. Claude did not check the elastic constants against the paper itself. B, C′ and A were not given as reference values, so they aren't compared.

### Things caught along the way
- The task asked for `NOTES.md`. The project log was then `PROJECT_A_NOTES.md` (named in `CLAUDE.md`), so this session was first written there. Later the same day it was renamed to `NOTES.md` (see "Things I caught" below).
- The reference a₀ = 2.8553 Å and E_coh = −4.1224 eV/atom match our own Session 1 output digit for digit. Session 1 recorded the published values only as "≈ 2.855 Å" and "≈ −4.122 eV/atom". Someone should confirm the 4-decimal values in the paper (Table) itself, so that we aren't comparing our result with itself.
- The potential file header cites Mendelev 2003 but doesn't say *which* of the paper's potentials it is ("potential 2" isn't written in the file). That it's potential 2 comes from the task description, not from the file.
- The default `log.lammps` is git-ignored, so the Session 1 evidence `log.lammps` in the project root stays on disk but isn't in git. Check 1 reproduces it exactly, and the new run is in git as `checks/01_lattice/log.lattice`.
- The settings not given in the task (CG minimiser, tolerances 1e-12/1e-12, `vmax 0.001`) were copied from the 20 Oct input, not newly chosen.

### Standing rules updated (later, 27 Oct)
- **What:** rewrote `CLAUDE.md` with Ansh's standing rules. The new rules: log every prompt verbatim in `PROMPTS.md` before starting work; record every result in `NOTES.md`; give a time estimate and run in the background with progress logs in `logs/` for runs over 2 minutes; commit locally after each step and never push. The older rules (conda env, ask before physics choices, numbers must be real) are kept.
- **Created** `PROMPTS.md`, with today's two prompts verbatim. The first one was logged after the fact, and its time (~19:35) comes from file timestamps.
- **Renamed** `PROJECT_A_NOTES.md` → `NOTES.md` (with `git mv`, so the history is kept), so the notes file has the name used in Ansh's prompts and in `CLAUDE.md`.

## Session 4 — Tue 27 Oct 2026: clean-up, potential identity, first screw dislocation cell

### Part A — clean-up and potential identity check
- **Project potential (Ansh's decision):** Mendelev 2003, potential 2, is the v0.1 potential. Proville 2012 comes in v0.3. See "Decisions so far".
- **References at paper precision:** the a₀/E_coh references in Session 3 are now 2.855 Å and −4.122 eV/atom (Mendelev et al. 2003). The 4-decimal values were our own 20 Oct output. Both still pass, at 0.011 %.
- **Old file name:** CLAUDE.md never mentioned `PROJECT_A_NOTES.md`. The three mentions left in NOTES.md all describe the rename itself, so they stay as history.
- **Is our potential file really "potential 2"? Yes.**
  - *Plain words:* we downloaded the official copy from the NIST potentials repository and compared every number in its tables with our file. All 30,000 tabulated values are identical. Only the free-text comment lines differ.
  - *Detail:* `tools/compare_eamfs.py ours other` compares the grid line (Nρ = 10000, dρ = 0.03, Nr = 10000, dr = 0.00053, cutoff 5.3 Å) and the tables F(ρ), ρ(r), r·φ(r).

| NIST file (saved in `potentials/reference/`) | Grid | Values | Verdict |
|---|---|---|---|
| `Fe_2_nist_v1.eam.fs`: entry `2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2`, version 1, header "Potential #2", dated 10 Jun 2007 | same | 30000/30000 identical (max diff 0) | **MATCH** |
| `Fe_2_nist_v3.eam.fs`: same entry, version 3, dated 22 Feb 2009 | same | differ (mass 55.85 vs 55.845; up to ~5 % in r·φ, ~3 % in ρ) | no match |
| `Fe_5_nist_v1.eam.fs`: potential #5 (negative control) | different (cutoff 6.0 Å) | — | no match |

Download: `curl -O https://www.ctcms.nist.gov/potentials/Download/2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2/1/Fe_2.eam.fs` (and `/3/`, and `...Fe-5/1/Fe_5.eam.fs`).

### Part B — one straight 1/2⟨111⟩ screw dislocation: build and relax (no stress applied)
- **Plain words:** a screw dislocation is a line defect. Around it the crystal's atomic planes form a spiral ramp, like a parking-garage ramp, rising by one Burgers vector b per turn. We put one straight screw dislocation in the middle of a thick crystal slab. The slab repeats endlessly along the line and along the glide direction, and has free top and bottom surfaces. Then we let the atoms settle into their lowest-energy positions.
- **Cell (frame relabelled, see below):** x = [1 1 1] (line, periodic), y = [−1 1 0] (glide-plane normal, free surfaces), z = [−1 −1 2] (glide direction, periodic). Dot products: x·y = 1.8e−17, x·z = 2.2e−17, y·z = 2.5e−17 (zero to round-off), right-handed.
- **Size:** Lx = 14.8366 Å (3 × a√3 = 6b), Ly = 149.4061 Å (37 × a√2), Lz = 146.8746 Å (21 × a√6). **27,972 atoms** (expected 25,000–35,000). a = 2.8553 Å as specified (the full relaxed value 2.855325 Å differs by 9×10⁻⁶ relative, which is negligible).

**How the mismatch at the periodic boundary is handled (simulation-setting choice).**
- *The problem, in plain words:* the textbook screw displacement u = (b/2π)·atan2 is the field of one dislocation in an infinite crystal. When the glide direction is made periodic, the left and right edges of the box don't fit together: the atoms are shifted by different amounts at different heights.
- *Options considered:*
  1. **Periodic array of dislocations (PAD) with a box tilt of b/2. Chosen (recommended).** Use the exact field of an infinite row of screws spaced Lz apart, u_x = (b/2π)·arg sin(π[(z−z0) + i(y−y0)]/Lz). This is the atan2 field plus all its periodic images, and it equals atan2 near the core. This field is shifted by exactly b/2 between the two box edges, in both the upper and the lower half-crystal. A box tilt xz = b/2 absorbs that shift exactly, so the cell is perfectly periodic and symmetric about the glide plane. Method: PAD model, Osetsky & Bacon, *Modelling Simul. Mater. Sci. Eng.* 11, 427 (2003). Checked numerically: shift = b/2 (mod b) to 1.6×10⁻¹¹ Å at every height.
  2. Plain atan2 with a linear correction and no tilt. This puts all the mismatch in one half-crystal, which gives a one-sided shear stress of ~1 GPa that pushes on the dislocation. Rejected.
  3. Plain atan2, and let the relaxation fix the seam. This leaves a planar defect at the box edge. Rejected.
- *Consequence:* LAMMPS can only tilt the box along x, so the line had to lie along x instead of z. Only the axis names change. Ansh's z = [111] is our x, Ansh's x = [1 1 −2] is our −z, and y is unchanged. In our frame the glide (Schmid) stress for later steps is σ_xy (normal y, b along x).
- *A side effect of any single-dislocation cell that is periodic along the glide direction:* each half-crystal carries a uniform shear strain of about ∓b/(2Lz). Measured after relaxation: γ_xz = −0.0096 (upper half) and +0.0097 (lower half). This is a *non-glide* stress, opposite in the two halves, and it cancels at the core by symmetry. Estimate: σ_xz ≈ ±0.69 GPa, using μ₁₁₁ = (C11−C12+C44)/3 = 71.44 GPa from our elastic constants (rough isotropic-in-plane estimate, not computed by LAMMPS). See "Things I caught".

**Easy-core position.** Two cells were built, with the core at the centre of an "up"-pointing or a "down"-pointing triangle of [111] columns. For a given sign of b, one of them is the easy core and the other the hard core.
- *down* (core at y0 = 74.0301, z0 = 72.2716 Å): the core **stays** at its triangle, so this is the easy core. This is the result cell.
- *up*: the core **moves** one triangle (~1.3 Å) into the neighbouring easy position. So the hard core is not stable with this potential. The end state has the same core structure. Its total energy differs by 0.0004 eV, because the core sits at a slightly different place in the finite cell.

**Relaxation:** CG, tolerances 1e-12/1e-12 (same as `checks/`), box fixed, `boundary p s p`. Run-time estimate before running: ≤ 3.5 min per run (upper bound at 10,000 iterations; 0.0105 s per force evaluation on 8 MPI ranks). Actual run time: 35 s (up) and 30 s (down). Converged in 185 and 169 iterations (energy criterion); final max force 1.6e−5 and 2.3e−5 eV/Å.

**Checks (easy-core cell, `down`):**

| Check | Computed | Expected / reference | Pass |
|---|---|---|---|
| (1) b = a√3/2 | 2.4728 Å | 2.473 Å (Ansh) | Pass |
| (2) smallest interatomic distance, after insertion | 2.3675 Å (up cell: 2.3313 Å) | > ~2.2 Å; bulk 1st-neighbour 2.4728 Å | Pass |
| (2) smallest interatomic distance, after relaxation | 2.3963 Å (both cells) | > ~2.2 Å | Pass (3 % below bulk, compressed core) |
| (3) energy per atom, far from core (r > 40 Å) and surfaces (> 15 Å); 15,480 atoms | mean −4.12229, min −4.12244, max −4.12207 eV/atom | −4.122 eV/atom (Mendelev 2003); our bulk −4.12244 | Pass (0.004 % vs paper) |
| (4) DD map: the three central-triangle |DD| | 0.3335 b, 0.3333 b, 0.3333 b | b/3 for a compact core | Pass |
| (4) three-fold symmetry of |DD| (within 10 Å) | max deviation 0.0010 b | 0 | Pass |
| (4) [−110] diad symmetry of |DD| (non-degenerate test) | max deviation 0.0000 b | 0 for non-degenerate, ≠ 0 for degenerate | Pass |

- The far-field atoms are 0.00015 eV/atom above bulk. This shrinks with distance from the core: −4.12223, −4.12228, −4.12233 eV/atom in rings 40–50, 50–60, 60–70 Å. What's left matches the order of the PAD shear-strain energy above (estimate ½μγ² ≈ 2.4×10⁻⁴ eV/atom, uniform-strain estimate).
- **Core structure:** the DD map (`figures/core_ddmap.png`) shows a **compact, non-degenerate core**. It spreads equally on three {110} planes, with three-fold and diad symmetry. Ansh's expectation was this core for this potential; Claude did not check it against the literature.
- **Files:** `sims/01_screw_cell/build.py` (builder, PAD field), `in.relax`, `analyze.py` (energy, strain, DD map and symmetry); logs `log.relax_down`, `log.relax_up`; progress `logs/screw_relax_*.log`; results `relaxed_down.data` (the cell for the next step) and `relaxed_up.data`; figures `figures/core_ddmap.png` (easy core) and `figures/ddmap_up_start.png` (hard-core start, after relaxation).
- **"Failed with exit code 1": not a simulation failure.** Both LAMMPS runs ended with exit code 0 (see `logs/screw_relax_*.log`). The 1 came from the last line of Claude's one-off shell command, `tail -2 file1 file2`: GNU `tail` rejects that old-style option when given more than one file (`tail -n 2` is correct). Reproduced and confirmed. While checking, Claude also fixed a latent bug in `tools/run_with_progress.py`. It created `logs/` only when started from the project root; it now creates `<project>/logs` from any folder (tested from `sims/01_screw_cell`).
- **Commands:**
```
cd sims/01_screw_cell
python build.py down screw_down.data            # and: up screw_up.data
python ../../tools/run_with_progress.py screw_relax_down 10000 -- mpirun -np 8 lmp -in in.relax -var tag down -log log.relax_down -screen none
python analyze.py down ../../figures/core_ddmap.png
```

## Session 5 — Tue 27 Oct 2026: Peierls stress at χ = 0

### 1. Peierls stress in the 146.87 Å cell (`sims/02_peierls_chi0/`)
- **Plain words:** the Peierls stress is the smallest shear stress that makes a straight dislocation glide at 0 K, with no help from heat. We push the top surface of the slab one way and the bottom surface the other way, along the dislocation line. We raise the push in small steps and let the atoms settle after each one. Below the Peierls stress the core only leans a little. Above it the core slides from one valley of the crystal to the next and doesn't stop.
- **Method (Ansh's choice, see "Design choices"):** glide stress σ_xy applied by surface forces (`fix addforce`, with `fix_modify energy yes`). The forces act on a 2-plane slab at each free surface: top +F along x, bottom −F along x.
  - **Slabs:** 756 atoms each, two (−110) planes of 378 atoms. Top planes at y = 145.395 and 147.412 Å; bottom planes at y = 0.037 and 2.051 Å. Plane spacing d110 = 2.019 Å, so the nominal thickness is 2·d110 = 4.04 Å.
  - **Hand check:** A = Lx·Lz = 2179.12 Å². 1 GPa·Å² = 0.006242 eV/Å. Total force per surface at 1 GPa = 13.601 eV/Å (expected ~13.6). Per atom 0.017991 eV/Å per GPa.
  - **Why `fix_modify energy yes` matters:** the minimiser lowers an energy and follows forces. If the applied force's work (−F·x) weren't in that energy, the forces wouldn't be the slope of the energy, so the line searches and the convergence test would be wrong.
- **Core tracker:** the excess-energy-weighted centroid along z of atoms within 4 Å of the glide plane (y0 = 74.03 Å) with excess energy > 0.01 eV. The excess is measured from the bulk value −4.1224 eV/atom. It is a circular (periodic) mean along z, unwrapped step by step. At σ = 0 it gives z = 72.2716 Å, exactly the inserted core position (z0 = 72.2716 Å).
- **Ramp:** 0 → 1.6 GPa in 50 MPa steps, each step starting from the previous relaxed state. CG, 1e-12/1e-12, cap 10,000 iterations. The ramp stops at the first step where the core moves at least one jump (2.331 Å) and is still moving. Then 10 MPa refinement from the last stable state.
  - *Runaway guard:* a step is stopped (`fix halt`) once the core has moved more than 3 jumps (6.99 Å) within that step. Such a step is marked "not converged (halted)".
  - *What "every step converged" means here:* a step above the Peierls stress can never converge, because in a cell periodic along the glide direction the dislocation keeps gliding. So the convergence condition was applied to the steps below the Peierls stress.
- **Run:** estimate ~5 min (upper bound ~15 min); actual 218 s for 31 steps. Each stable step took 166–204 CG iterations and converged on the energy criterion, with fmax ≤ 5.1e−5 eV/Å.
- **Result:**

| Quantity | Computed | Reference | Difference |
|---|---|---|---|
| Last stable stress | 1.280 GPa (core +1.098 Å from start, converged) | | |
| First moving stress | 1.290 GPa (core ran +6.99 Å within the step, halted) | | |
| **Peierls stress, χ = 0 (146.87 Å cell)** | **1.280 – 1.290 GPa** (midpoint 1.285) | 1.21 GPa (Chaussidon, Fivel & Rodney, *Acta Mater.* 54, 3407 (2006), Mendelev potential 2, free boundaries, 1210 MPa at χ = 0, text Section 4; read by the planning chat 28 Oct; corrected from the earlier ~1.25 GPa restated by Gilbert et al. 2013) | +5.8 % to +6.6 % (midpoint +6.2 %) |

- **Pass/fail:** not decided yet (Ansh).
- **Sign check:** under +σ_xy the core moves toward **+z** (= [−1 −1 2]). That was predicted before the run from the inserted field. The slipped region (top shifted +x relative to bottom by b) lies on the −z side of the core, and +σ_xy pushes the top toward +x, so the slipped region grows and the core moves to +z. It also matches the work term: the total energy falls as the core advances. **Match.**
- **Before the Peierls stress**, the core leans forward smoothly, by +1.10 Å at 1.28 GPa (less than half a jump), and it stays on its glide plane: core y between 73.68 and 73.76 Å over the whole ramp. Some steps show small extra moves of 0.07–0.10 Å (at 0.15, 0.30, 0.45, 0.85, 1.10, 1.20 GPa). These are probably tracker noise from atoms crossing the 0.01 eV threshold, not physical jumps; all are far below one jump.
- **Files:** `tools/peierls.py` (driver), `sims/02_peierls_chi0/in.step` (one step), `groups.lmp` (slab atom IDs), `steps/log.*` (one LAMMPS log per step; step data files are git-ignored), `data/peierls_chi0.csv`, `figures/peierls_chi0.png`, progress `logs/peierls_chi0.log`.
- **Command:** `cd sims/02_peierls_chi0 && python ../../tools/peierls.py ../01_screw_cell/relaxed_down.data 74.0301 peierls_chi0`

### 2. Cell-size convergence test (2× glide length), `sims/03_peierls_chi0_2x/`
- **Interruption (27 Oct, ~20:32):** the Ubuntu windows closed while the 2× test was starting. Nothing had been written yet (empty folder, no process left), and the χ = 0 result above was already committed. The restart at 20:42 with `run_2x.sh` crashed WSL as well (builder out of memory); the test was done in Session 6.

## Session 6 — Tue 27 Oct 2026 (evening): crash-safe runs and the 2× cell-size test

- **Crashes:** the Ubuntu window closed by itself three times (~20:35, ~20:43, ~20:47; the first time WSL itself restarted, the third with nothing running). **Most likely cause of the first two: the 2× cell builder ran the laptop out of memory.** `build.py` made the whole cubic grid at once, ≈ 3.5 GB per array and at least 3 arrays for the 294 Å cell, on a 7.6 GB laptop. `run_2x.sh` was written at 20:42, one minute before the second crash. The third crash is unexplained (Windows Insider build 10.0.26300, WSL 2.7.3). Ubuntu now runs from a PowerShell window (`wsl -d Ubuntu`).
- **New launch rule (in `CLAUDE.md`):** every LAMMPS run starts through `tools/launch_safe.sh`: detached (`setsid nohup`), output only to `logs/<run>.out`, `OMP_NUM_THREADS=1`, at most cores − 1 = 7 MPI ranks, PID in `logs/<run>.pid`.

### 1. State before starting
- `sims/03_peierls_chi0_2x/` held only `run_2x.sh` and an empty `run.out`: no data, no logs. No LAMMPS process was running.
- The old runs used `mpirun -np 8` (all 8 cores), with OMP_NUM_THREADS not set. LAMMPS ran with `-screen none`; the Peierls driver printed ~2 lines per step (67 lines per ramp).

### 2. Fixes
- **Builder memory:** `sims/01_screw_cell/build.py` now builds the lattice one plane at a time. It gives a byte-identical 146.87 Å cell (`cmp` of the data file, and identical reference positions). Peak memory is 35 MB.
- **Launcher bug caught during the test:** plain `nproc` obeys `OMP_NUM_THREADS`, so it reported 1 core and gave NP = 0. The launcher now uses `nproc --all` (8). The bad run was killed and redone.
- `tools/peierls.py` now uses NP (7) by default instead of 8.

### 3. Crash-safe test: re-run of the 146.87 Å relaxation (`sims/00_crashsafe_test/`)

| Quantity | Re-run (7 ranks, detached) | Original (8 ranks, S4) | Match |
|---|---|---|---|
| E after insertion | −114848.093067 eV | −114848.093067 eV | identical |
| E after relaxation | −114849.235516 eV | −114849.235516 eV | identical |
| dmin after relaxation | 2.3963 Å | 2.3963 Å | identical |
| Iterations | 170 (energy tol.) | 169 | round-off from a different rank count |
| Minimiser time | **3.8 s** | 23.5 s | — |

- The log is complete ("Total wall time" line). The process ran in its own session with no terminal, and nothing reached the screen (`logs/crashsafe_test.out` is empty).
- **7 ranks were 6× faster than 8:** using every core made the laptop compete with Windows for CPU.

### 4. 2× cell (`sims/03_peierls_chi0_2x/`, SCREW_NZ = 42): build checks

| Check | 2× cell (computed) | 146.87 Å cell (S4) | Expected | Pass |
|---|---|---|---|---|
| b | 2.4728 Å | 2.4728 Å | 2.473 Å | Pass |
| Box Lx × Ly × Lz | 14.8366 × 149.4061 × 293.7492 Å | Lz 146.8746 | Lz ~294 Å | Pass |
| Atoms | 55,944 | 27,972 | ~56,000 (exactly 2×) | Pass |
| dmin after insertion / relaxation | 2.3675 / 2.3963 Å | 2.3675 / 2.3963 | > ~2.2 Å | Pass |
| Relaxation | 286 it, energy tol., fmax 2.6e−5 eV/Å, 15 s | 169 it | converged | Pass |
| Far-field energy (r > 40 Å, > 15 Å from surfaces; 37,404 atoms) | mean −4.12238 (min −4.12244, max −4.12214) eV/atom | −4.12229 | −4.122 (Mendelev 2003); our bulk −4.12244 | Pass |
| Built-in shear γ_xz, upper / lower half | −0.00561 / +0.00557 | −0.0096 / +0.0097 | "about half" | See note |
| Core: central-triangle DD; 3-fold; diad | 0.3334, 0.3333, 0.3333 b; 0.0006 b; 0.0000 b | same | b/3; 0; 0 | Pass |

- **γ note:** the measured γ is 58 % of the small cell's, not 50 %. Independent check: fitting the *unrelaxed* isotropic PAD field in the same band (30–60 Å from the core) gives 0.00955 (small) and 0.00533 (2×), the same ratio of 0.56. So the extra 8 % comes from the measuring band, which still sits inside the core's own field in a 294 Å period. The uniform PAD shear b/(2Lz) itself is halved exactly (0.00842 → 0.00421).
- The far-field energy is closer to bulk than in the small cell (−4.12238 vs −4.12229), as expected with less built-in shear.

### 5. Peierls stress in the 2× cell
- **Run:** the same driver and input as `sims/02_peierls_chi0`: slabs of 1,512 atoms (2 planes of 756), A = 4358.23 Å², 27.202 eV/Å per surface at 1 GPa, 0.017991 eV/Å per atom per GPa (same per-atom force as the small cell). The estimate was 7–8 min (upper bound ~15); the actual time was 368 s for 31 steps.
- All 30 steps below the runaway converged (energy criterion, fmax ≤ 4.9e−5 eV/Å). The core moved toward +z under +σ_xy, the same sign as the small cell.
- **Unexpected event at 1.280 GPa (fine step, from the 1.270 state):** the core slid **+2.93 Å** (more than one jump, 2.331 Å) and then **stopped**, converged at 430 iterations. The tracked core y rose from 73.74 to 74.19 Å (+0.45 Å). At 1.290 GPa it stayed put (+0.005 Å). This did not happen in the small cell. ~~The pair energy is +0.84 eV higher, a sign of a different core.~~ **Corrected (Ansh):** +0.84 eV is the normal elastic rise per 10 MPa step (1.26→1.27: +0.817 eV, 1.27→1.28: +0.840 eV, 1.28→1.29: +0.840 eV, from `data/peierls_chi0_2x.csv`). The real clues are the non-integer move (2.93 Å = 2.331 + 0.60 Å) and the height change. Resolved by the DD maps in Session 7.
- The runaway step was the **coarse 1.300 GPa** step, which started from the 1.250 GPa state (+7.04 Å, halted). 1.300 GPa was never applied to the post-jump state.

| | 146.87 Å cell | 293.75 Å cell (2×) | Difference (midpoints) |
|---|---|---|---|
| Driver's bracket (runaway = "moves ≥ 1 jump and keeps going") | 1.280 – 1.290 GPa | 1.290 – 1.300 GPa | +0.8 % |
| First move of ≥ 1 jump from the original easy-core state | 1.280 – 1.290 GPa | 1.270 – 1.280 GPa | −0.8 % |
| Widest spread between the brackets | | | ±1.6 % |
| Reference (Chaussidon, Fivel & Rodney 2006, Section 4, read by the planning chat; corrected 28 Oct from ~1.25) | 1.21 GPa | 1.21 GPa | |

- **Acceptance test (< 3 %): met under either reading** (difference at most 1.6 %). **Ansh's decision (Session 7):** the Peierls stress is the first stress at which the core leaves its starting valley by one jump or more, so the 2× cell gives 1.270–1.280 GPa, a difference of 0.8 %. The 146.87 Å cell is accepted.
- **Files:** `sims/03_peierls_chi0_2x/build_2x.sh`, `log.relax_down`, `relaxed_down.data`, `in.step`, `groups.lmp`, `steps/log.*`; `data/peierls_chi0_2x.csv`; `figures/peierls_chi0_2x.png` (both cells), `figures/core_ddmap_2x.png`; logs `logs/build_2x.out`, `logs/screw_relax_2x.log`, `logs/peierls_chi0_2x.log`, `.out`; test `sims/00_crashsafe_test/log.relax_down`, `logs/crashsafe_test*.{out,log}`.
- **Commands:**
```
cd sims/03_peierls_chi0_2x
../../tools/launch_safe.sh build_2x ./build_2x.sh
../../tools/launch_safe.sh peierls_chi0_2x python ../../tools/peierls.py relaxed_down.data 74.0301 peierls_chi0_2x
ps -p $(cat ../../logs/peierls_chi0_2x.pid)      # is it still running?
```

## Session 7 — Tue 27 Oct 2026 (evening): what the core did at 1.280 GPa in the 2× cell

- **Plain words:** a "differential displacement" (DD) map draws an arrow between neighbouring atom columns; the arrow length shows how much the two columns have slid past each other along the line. A normal (compact, easy) core shows three equal arrows of b/3 around one triangle of columns. We drew this map for the 2× cell just before (1.270 GPa) and after (1.280 GPa) the odd move.
- **Answer: at 1.280 GPa the core is no longer the compact easy core. It is spread around a single atom column**, which is the shape of a core centred on one column (as in the "split" core). Gilbert et al. 2013 report a metastable split core for the Mendelev potential, now known to be an artifact (read by the planning chat (Claude in claude.ai)). In DFT the split core has a high energy, near or above the hard core, so it is not a resting state (Dezerald, Ventelon, Clouet, Denoual, Rodney, Willaime, "Ab initio modeling of the two-dimensional energy landscape of screw dislocations in bcc transition metals", HAL record hal-02311245; read by the planning chat). Not yet compared with a published split-core map. It is still the same shape at 1.290 GPa, where it was stable.

| State (2× cell) | Where the core is | Arrows of the triangle holding b | Sum of arrows around the top 3 columns |
|---|---|---|---|
| σ = 0 (relaxed) | inserted triangle | 0.3333, 0.3334, 0.3333 b | 1.10 / 1.10 / 1.10 b (equal: compact core) |
| 1.270 GPa | inserted triangle (no move) | 0.313, 0.333, 0.354 b | 1.21 / 1.09 / 1.00 b (compact, skewed by the stress) |
| **1.280 GPa** | **centred on column C at dz = +3.50 Å (1.5 jumps), dy = +0.67 Å** | 0.475, 0.279, 0.246 b (triangle next to C) | **1.58** / 0.96 / 0.96 b (one column dominates) |
| 1.290 GPa | same as 1.280 | 0.475, 0.279, 0.246 b | 1.58 / 0.96 / 0.96 b |

- **Column C's six bonds at 1.280 GPa:** 0.486, 0.475, 0.246, 0.230, 0.080, 0.063 b. C sits halfway between two easy-core sites on the glide path (z = 2.33 and 4.66 Å), on the row of atoms just above the glide plane (+0.67 Å). That fits the tracker's 2.93 Å move and +0.45 Å height change.
- **Small cell (`sims/02_peierls_chi0`), all 31 steps checked with the same DD method:** in every one of the 29 stable steps the core stays in its starting triangle. The tracked height varies only between 73.678 and 73.761 Å (at most 0.035 Å per step), and the largest move per step is 0.188 Å. The two runaway states (1.290 and 1.300 GPa) sit exactly 3.000 jumps along z on the same plane (where the halt guard stopped them), with a compact triangle core. **No partial jump and no height change before the runaway.** In the 2× cell the 1.300 GPa runaway also stayed on the plane (3.000 jumps); only the 2× 1.280 GPa step ended up on a column.
- **Plots:** `tools/plot_peierls.py` now has a second panel with the tracked core height; `figures/peierls_chi0.png` and `figures/peierls_chi0_2x.png` were redrawn.
- **Files:** `tools/ddmap_data.py` (DD map + core location from any step data file), `figures/core_ddmap_2x_1270.png`, `figures/core_ddmap_2x_1280.png`.
- **Commands:**
```
cd sims/03_peierls_chi0_2x
SCREW_NZ=42 python ../../tools/ddmap_data.py steps/fine_1.280.data ../../figures/core_ddmap_2x_1280.png "2x cell (293.75 A), sigma_xy = 1.280 GPa, relaxed"
cd ../01_screw_cell   # small cell: SCREW_NZ=21, step files in ../02_peierls_chi0/steps/
```

## Session 8 — Tue 27 Oct 2026 (late evening): does an extra xz box tilt give a clean σ_xz? (no dislocation)

- **Plain words:** for angles χ ≠ 0, Ansh chose to add the second shear component σ_xz by tilting the box a little more along xz, with σ_xz = G·γ and G = (C11 − C12 + C44)/3. Before using it on a dislocation, we tested it on a perfect crystal: tilt the box, let the atoms settle, and read the full stress. **Result: σ_xz comes out exactly as expected, but the tilt also creates normal stresses σ_yy and σ_zz of about 44 % of σ_xz each. This is real physics, not a bug: iron is elastically anisotropic (A = 2.36), and in a frame aligned with [111] a shear in xz is coupled to normal stresses. So the test fails Ansh's "< 1 %" criterion, and the work stopped here for Ansh's decision.**

### 1. Rotated stiffness tensor (calculation, `checks/03_tilt_xz/rotate_C.py`)
- Input: our C11 = 243.3297, C12 = 145.0091, C44 = 115.9907 GPa (`checks/02_elastic`, e = 0.001), rotated to x = [111], y = [−110], z = [−1−12].
- **C'_55 (σ_xz per γ_xz) = 71.4371 GPa**, equal to (C11 − C12 + C44)/3 = 71.4371 GPa. Pass.
- Couplings of γ_xz to other stresses: **C'_25 = +31.504 GPa (σ_yy), C'_35 = −31.504 GPa (σ_zz)**, i.e. ±44.1 % of C'_55. C'_15 (σ_xx) = C'_45 (σ_yz) = C'_65 (σ_xy) = 0.
- *Hand check:* for a cubic crystal in a [111]-aligned frame the coupling is (C11 − C12 − 2C44)/(3√2) = (243.33 − 145.01 − 231.98)/4.243 = −31.50 GPa in magnitude. It vanishes only for an isotropic crystal (C11 − C12 = 2C44). Same size as computed.

### 2. LAMMPS test (`checks/03_tilt_xz/in.tilt`, `run_all.sh`, crash-safe launcher)
- **Block:** perfect bcc, fully periodic, a = 2.8553 Å, 5 × 10 × 4 repeats of a√3 × a√2 × a√6: Lx = 24.7276, Ly = 40.3800, Lz = 27.9761 Å, **2400 atoms** (expected 2400.0), smallest distance 2.4728 Å (= a√3/2).
- **Zero strain:** σ_xx = σ_yy = σ_zz = −0.0047 GPa, shear 0. That residual is because a = 2.8553 is our relaxed 2.85532 rounded; estimate (C11 + 2C12)·Δa/a ≈ 533 GPa × 8.7×10⁻⁶ = 0.0046 GPa. Stresses below are changes from this state.
- **Strained:** tilt xz = γ·Lz, atoms relaxed at fixed box (CG 1e-15, fmax ≤ 9×10⁻¹⁴ eV/Å). Sign convention: LAMMPS pressure P_ij = −σ_ij (tension positive).

| γ | Δσ_xz (GPa) | G·γ (GPa) | Diff. | Δσ_yy (GPa) [pred.] | Δσ_zz (GPa) [pred.] | Δσ_xx (GPa) | Δσ_yz, Δσ_xy |
|---|---|---|---|---|---|---|---|
| +0.005 | +0.35773 | +0.35719 | +0.15 % | +0.15859 [+0.15752] | −0.15469 [−0.15752] | +0.00277 (0.77 %) | 0 |
| +0.010 | +0.71638 | +0.71437 | +0.28 % | +0.31917 [+0.31504] | −0.30335 [−0.31504] | +0.01110 (1.55 %) | 0 |
| −0.005 | −0.35690 | −0.35719 | −0.08 % | −0.15656 [−0.15752] | +0.16060 [+0.15752] | +0.00275 (0.77 %) | 0 |

- **σ_xz: pass** (within 0.3 %, linear, opposite sign at −0.005).
- **Other components: fail.** σ_yy and σ_zz are ±43–45 % of σ_xz and flip sign with γ, as predicted. The odd part at ±0.005 agrees with the tensor to 0.04 % (σ_yy: 0.15758 vs 0.15752; σ_xz: 0.35731 vs 0.35719). σ_xx is the same sign for ±γ, so it is a second-order (γ²) effect: 0.77 % at |γ| = 0.005, 1.55 % at 0.010.
- **In the dislocation cell** (calculation only, `slab_case.py`): there the y faces are free, so σ_yy relaxes away. A pure tilt then gives σ_zz = −0.644 σ_xz, σ_xx = −0.150 σ_xz, and an effective G = 68.24 GPa (4.5 % below 71.44).
- **Files:** `checks/03_tilt_xz/` (`rotate_C.py`, `in.tilt`, `run_all.sh`, `compare.py`, `slab_case.py`, logs `log.tilt_*`); `logs/tilt_xz.out`.
- **Commands:** `cd checks/03_tilt_xz && python rotate_C.py && ../../tools/launch_safe.sh tilt_xz ./run_all.sh && python compare.py && python slab_case.py`

## Session 9 — Tue 27 Oct 2026 (night): preparation checks for χ ≠ 0 (Ansh away; perfect crystals and post-hoc analysis only)

- **Plain words:** four checks before any χ ≠ 0 run:
  1. **Does our χ = 0 loading carry a hidden extra stress?** To first order, no. At the Peierls stress a small second-order leftover of ~3 % appears.
  2. **Option 2 (cancel the tilt's normal stresses with small box stretches):** it works on a perfect crystal once the strains get one measured correction.
  3. **Twinning vs anti-twinning:** which side of χ = 0 is which, worked out from geometry.
  4. **A core tracker that follows the core:** written and tested. It sees moves off the plane, but under stress its sub-Å position depends on the window size.
- No dislocation run at χ ≠ 0 and no angle sweep was started. **Option 2 is tested, not yet decided (Ansh).**

### 1. σ_xy coupling and the χ = 0 loading (`checks/04_pure_shear/elastic_calc.py`, `in.block`, `run_all.sh`, `run2.sh`)
- **xy column of the rotated stiffness:** C'_46 = +31.504 GPa (γ_xy → σ_yz), C'_66 = 71.437 GPa; C'_16 = C'_26 = C'_36 = C'_56 = 0. As expected by symmetry.
- **Check, periodic block, tilt γ_xy = 0.005** (2400 atoms, changes from the γ = 0 state): Δσ_xy = 0.35731 (predicted 0.35719), Δσ_yz = 0.15758 (predicted 0.15752) GPa. Others ≤ 0.0028 GPa (second order). Pass.
- **Slab with free y faces under surface-force σ_xy (linear calculation):** σ_yy = σ_yz = 0 at the faces, so the coupling is absorbed by a free strain γ_yz = −0.0055 per GPa (the top face slides slightly along z). **All non-glide stresses are zero to first order.** The effective glide stiffness is C'_66 − C'_46²/C'_44 = 60.85 GPa.
- **Check, perfect slab with surface forces** (`boundary p s p`, same 2-plane slabs, `fix_modify energy yes`). The interior stress is the average of stress/atom over atoms within 8 Å of mid-plane, minus the unloaded slab; this averaging gives the global pressure exactly on the periodic block.

| σ_xy applied (GPa) | interior σ_xy | σ_xz | σ_xx | σ_zz | σ_yy, σ_yz | largest other / σ_xy |
|---|---|---|---|---|---|---|
| 0.5 | 0.49999 | −0.0057 | +0.0050 | −0.0016 | ≤ 0.0001 | 1.1 % |
| 1.0 | 0.99984 | −0.0227 | +0.0201 | −0.0063 | ≤ 0.0001 | 2.3 % |
| −1.0 | −0.99984 | −0.0227 | +0.0201 | −0.0063 | ≤ 0.0001 | 2.3 % |
| **1.285** | **1.28467** | **−0.0376** | **+0.0334** | **−0.0107** | ≤ 0.0001 | **2.9 %** |

- The leftovers have the same sign for +σ and −σ and grow as σ² (×4 from 0.5 to 1 GPa), so they are **second-order** (finite strain, lattice rotation, non-linear elasticity) and **not a linear coupling**. Stresses are Cauchy stresses in the box frame.
- **Answer for the χ = 0 result (1.280–1.290 GPa):** the surface-force loading itself puts no linear non-glide stress on the core. At the Peierls stress, a second-order σ_xz ≈ −0.038 GPa (about −1.7° in MRSS angle) plus σ_xx ≈ +0.033 and σ_zz ≈ −0.011 GPa is present, measured in a perfect slab, not in the dislocation cell. It is the same in both cell sizes, so it doesn't affect the cell-size test. Separately, the already known PAD built-in shear (±0.69 GPa, opposite in the two halves, cancelling at the core) is still there. With the sign convention of item 3, the −1.7° points to the twinning side (inference, not tested).

### 2. Option 2: pure σ_xz from box strains (tested, not yet decided)
- **Strains per GPa of pure σ_xz** (compliance S = C'⁻¹, linear): ε_xx = 0, **ε_yy = −0.0027625, ε_zz = +0.0027625, γ_xz = +0.0164349**, γ_yz = γ_xy = 0. In the slab (free y faces) only ε_zz and γ_xz need to be set; ε_yy appears by itself. The calculation gives the same pure stress.
- **Linear strains, LAMMPS (fail):** σ_xz is +0.69 % (0.357) and +1.39 % (0.714) off. Other components: σ_zz up to 1.7 % / 3.3 % of σ_xz, σ_xx up to 2.5 %; same sign for ±0.357, so second order. The slab gives similar values (1.4 % / 2.9 %).
- **After one measured correction** (δε = S·(target − measured) for the block; for the slab, the 3×3 box-strain stiffness, `correct.py`): **pass.**

| Case | ε_xx | ε_yy | ε_zz | γ_xz | σ_xz vs target | largest other / σ_xz |
|---|---|---|---|---|---|---|
| block 0.357 | −0.00000969 | −0.00097404 | +0.00096011 | +0.00581600 | −0.02 % | 0.04 % |
| block 0.714 | −0.00003906 | −0.00192206 | +0.00186790 | +0.01152744 | −0.09 % | 0.16 % |
| slab 0.357 | −0.00000963 | (free) | +0.00096119 | +0.00582188 | −0.02 % | 0.03 % |
| slab 0.714 | −0.00003896 | (free) | +0.00187212 | +0.01155137 | −0.07 % | 0.13 % |

- For later: the linear values per GPa are above. The correction is ~1–2 % of the strains, plus a small ε_xx (−3.9×10⁻⁵ at 0.714 GPa). Not tested: the same strains on top of the PAD tilt and a dislocation.

### 3. Twinning vs anti-twinning (geometry only, `checks/05_twinning/twin_sense.py`, `symmetry_check.py`)
- **Convention:** the MRSS plane normal is n = cos χ ŷ + sin χ ẑ. The resolved shear along x on that plane is σ_xy n_y + σ_xz n_z = τ > 0, so **the crystal on the +n side is sheared along +x = +[111]** relative to the −n side.
- **Planes:** χ = +30° → **(−2 1 1)**, normal (y, z) = (+0.866, +0.500). χ = −30° → **(−1 2 −1)**, normal (+0.866, −0.500). Both contain [111].
- **Criterion (stacking):** {112} planes stack in a 6-layer sequence. Going one layer towards +n, the in-plane offset along b = a/2[111] is c·b with c = 1/3 or 2/3. The twin is the mirror stacking (−c per layer). Turning one into the other needs an extra shift of −2c (mod 1) per layer. The **twinning sense** is the short one, a/6⟨111⟩ (1/3 b); the opposite sense needs a/3⟨111⟩ (2/3 b) and is the **anti-twinning** sense.
- **Hand derivation Ansh can check, for (−2 1 1):** the lattice vector t = [0 1 0] joins two neighbouring layers (n·t = 1). Its component along b is (t·b)/|b|² = (1/2)/(3/4) = 2/3, so c = 2/3. The twin needs −4/3 ≡ −1/3, so twinning moves the +n side along **−[111]**. Our loading moves it along +[111], so it is **anti-twinning**. For (−1 2 −1): t = [0 1 0] again, n·t = 2, so use t = [0 0 −1]: n·t = 1, (t·b)/|b|² = (−1/2)/(3/4) = −2/3 ≡ 1/3, so c = 1/3. The twin needs −2/3 ≡ +1/3, so twinning moves the +n side along **+[111]**: **twinning**.
- **Checks of the criterion:**
  - It reproduces the textbook bcc twinning system **(112)[1̄1̄1]** (+n side of (112) moves along [1̄1̄1]).
  - Mapping (112)[1̄1̄1] onto our two planes with all 48 cubic symmetry operations gives the same two answers.
- **Result:** in our frame and loading, **χ = +30° (−211) is the anti-twinning sense, χ = −30° (−12−1) the twinning sense.**
- **Sources (updated 28 Oct):**
  - Gröger, Bailey & Vitek, *Acta Mater.* 56, 5401–5411 (2008), Part I (read by the planning chat).
  - Gröger, Racherla, Bassani & Vitek, *Acta Mater.* 56, 5412–5425 (2008), Part II: it defines χ < 0 as twinning and χ > 0 as anti-twinning for the nearest {112} plane, **which matches our convention** (read by the planning chat).
  - Christian & Mahajan, *Prog. Mater. Sci.* 39, 1–157 (1995) (read by the planning chat).
  - Gröger, Holzer & Kruml, *Comput. Mater. Sci.* 216, 111874 (2023): α-Fe twins only on {112} planes sheared in the twinning sense (read by the planning chat).
  - Hirth & Lothe, *Theory of Dislocations* (2nd ed., 1982): standard text, not consulted.
- **Second hand derivation (planning chat, 28 Oct):**
  - (h,k,l) → (−l,h,k) maps (112) to (−211) and [−1−11] to −[111]. So on (−211) twinning moves the +n side along −[111]; our loading moves it along +[111], hence **anti-twinning at χ = +30°**.
  - (h,k,l) → (−h,l,−k) maps (112) to (−12−1) and [−1−11] to +[111], hence **twinning at χ = −30°**.
  - Same as the stacking calculation and the 48-operation check above.

### 4. Core tracker that follows the core (`tools/track_core.py`, `tools/in.pe_dump`, `tools/run_track.sh`, `checks/06_tracker/test_tracker.py`)
- **Method:**
  - Per-atom energy comes from a LAMMPS `run 0` on each step's data file (64 runs, crash-safe).
  - Excess energy > 0.01 eV above −4.1224 eV is the weight (same as the old tracker).
  - Seed: z from the periodic mean over a band |y − y_prev| < 6 Å along the whole cell, y = y_prev.
  - Then the centroid inside a **circle of radius R = 6 Å** around the current estimate is computed repeatedly until it stops moving (< 10⁻⁵ Å).
  - Each step starts from the previous step's core, chained like the driver. Post-hoc only, nothing re-run.
- **σ = 0:** dy = +0.008 Å (small cell), +0.031 Å (2×); dz = 0.000. It doesn't depend on R. The old band tracker was off by −0.34 Å in y.
- **Small cell, all 31 steps:** same verdict as the old tracker. No step moves ≥ 1 jump before the runaway; the runaways are at +3.29 jumps (1.290 fine, 1.300 coarse). The new tracker stays within dy = +0.008…+0.12 Å, so the core stays on its plane. **But the pre-jump lean differs:** new z is 0.00 to 0.44 Å below the old one (largest near the Peierls stress). The old tracker's 0.07–0.10 Å jitter shows up in the new one too, at the same stresses, where the atom count in the window changes (108 → 114 → 120…).
- **2× cell, fine_1.270 → 1.280:** new tracker **dy = +1.173, dz = +3.702 Å (1.588 jumps)** at 1.280 GPa (old: +0.161 / +4.031). **Expected from the DD map: +0.67 / +3.50 Å: not reproduced in y (off by +0.50 Å) and +0.20 Å in z.** The energy centroid lands between column C (0.67, 3.50) and the Burgers triangle (2.02, 3.50); the core is spread and asymmetric.
- **Window sensitivity (2× cell):**

| State | R = 4 | R = 5 | R = 6 | R = 8 | R = 10 |
|---|---|---|---|---|---|
| σ = 0 (dy, dz) | +0.030, 0.000 | +0.031, 0.000 | +0.031, 0.000 | +0.031, 0.000 | +0.031, 0.000 |
| 1.270 GPa | +0.050, +0.366 | +0.086, +0.835 | +0.103, +0.666 | +0.080, +1.067 | +0.014, +1.204 |
| 1.280 GPa | +0.649, +3.810 | +0.882, +3.572 | +1.173, +3.702 | +0.905, +4.112 | +0.923, +4.176 |

- **Reading:**
  - Under stress, the energy centroid depends on the window by up to ~0.8 Å, so sub-Å "lean" values (from either tracker) aren't a sharp core position.
  - The new tracker does reliably detect jumps (≥ 2.33 Å) and moves off the plane (~1 Å; the old one saw only +0.16 Å).
  - The exact site comes from the DD core finder, which Ansh's decision already requires at the bracket steps.
  - R was not tuned to match the expected numbers.

## Session 10 — night 27→28 Oct 2026 (autonomous, Ansh asleep): χ-sweep gates and ML code

- **Plain words:**
  - The loading for χ ≠ 0 was calibrated on a perfect crystal and works: Gate A passed.
  - Before the sweep, the new code had to reproduce the χ = 0 result (Gate B). It did not: 1.250–1.260 instead of
    1.280–1.290 GPa. The reason is not a bug: near the Peierls stress the remaining energy barrier is so small that whether
    the minimiser hops over it depends on round-off details (here, 7 instead of 8 parallel processes).
  - As instructed, the sweep was **not** started.
  - Track 2 (ML code, tests, private README) is done.
- **Provisional decisions for this batch (Ansh, final approval pending):**
  - option-2 loading with the measured correction;
  - tracker for jumps and plane changes only, with DD maps for positions and shapes;
  - Peierls definition unchanged.
- All χ ≠ 0 material is labelled **provisional (option 2 not yet approved)**.

### Gate A — loading calibration on the perfect slab (`checks/07_chi_calibration/`): PASS
- **Loading per (χ, τ):**
  - surface forces for σ_xy = τ cos χ;
  - box strains ε_xx, ε_zz, γ_xz for σ_xz = τ sin χ, starting from the linear slab values (per GPa of σ_xz: ε_zz = +0.0027625, γ_xz = +0.0164349);
  - then Newton corrections from the measured interior stress (box strains via the slab box-strain stiffness, surface-force stress via its own residual).
- **v1** (τ = 1.0, 1.5): 22/22 calibrated points pass. My extra check of the interpolation rule (q₂τ² + q₃τ³) failed at τ = 2.4, χ = ±6°: σ_xz was 2.8 % off its small target (0.007 GPa, 0.3 % of τ). Kept as `calibration_v1.csv`.
- **v2** (used): calibrated at τ = 1.0, 1.5, 2.0, 2.4 for χ = −30…+30 in 6° steps plus 0 (44 points). **All pass**: max error σ_xy 0.013 %, σ_xz 0.096 %, others 0.082 % of τ (criteria 1 %, 1 %, 2 %).
- **Rule for other τ:** correction (calibrated − linear) = q₂τ² + q₃τ³ + q₄τ⁴, least squares per χ (`strain_model.json`). Checked without iteration at τ = 0.5, 1.25, 1.75, 2.2: **44/44 pass** (max σ_xz 0.875 %, others 0.24 % of τ).
- Typical calibrated strains at τ = 1.5 GPa: χ = +30°: ε_xx −0.000193, ε_zz +0.002158, γ_xz +0.012873, surface-force stress 1.30223 GPa (target σ_xy 1.29904); χ = −30°: −0.000166, −0.002009, −0.011898, 1.29670 GPa. Full table: `calibration.csv`.

### Gate B — χ = 0 regression through the new code (`tools/peierls_chi.py`, `tools/in.step_chi`, `sims/04_chi_sweep/chi0_nocorr/`): FAIL
- **Run:** χ = 0, `--nocorr`, i.e. no box strain and surface stress exactly τ, the same loading as Session 5. 28 steps, 205 s, 7 MPI ranks.
- **Result: 1.250 (last unmoved) – 1.260 GPa (first moved)**, against 1.280–1.290 GPa in Session 5.
  - At 1.260 GPa the core made one clean jump (DD: Burgers triangle at +1.000 jump, compact b/3 core) and **converged** there (312 iterations, energy criterion).
  - 1.300 GPa (coarse) ran away (halted after leaving the 6 Å window).
- **Diagnosis (analysis only, no extra run):**

| Step | E old (8 ranks) | E new (7 ranks) | ΔE (eV) | Iterations old/new | max \|Δr\| between states |
|---|---|---|---|---|---|
| coarse 0.500 | −114853.20865 | −114853.20866 | −0.000001 | 195/211 | 4.4×10⁻⁴ Å |
| coarse 1.250 | −114874.33044 | −114874.33044 | −0.000001 | 191/214 | 7.2×10⁻⁴ Å |
| fine 1.260 | −114874.73530 | −114875.40794 | **−0.673** | 166/312 | 14.8 Å (one jump) |

- **Reading:** up to 1.250 GPa both runs are the same (round-off differences only). At 1.260 GPa, from practically the same state under the same load, one CG minimisation stayed in the starting valley and the other crossed the small remaining barrier into the next valley. Both end states are converged minima.
- ~~So the static bracket carries a minimiser-path uncertainty of at least ~0.03 GPa (~2 %), and that applies to the accepted 1.280–1.290 GPa as well.~~ **Withdrawn (Session 11; Ansh, 28 Oct): wrong.** There is no minimiser-path dependence. The cause was the `change_box … remap` y-stretch in `in.step_chi` (Session 11 §2).
- **Track 1 stopped here.** Possible next steps, for Ansh to choose (not tried): (i) a smaller maximum minimiser step (`min_modify dmax`); (ii) run each bracket step from several slightly perturbed starts, or with both rank counts; (iii) a stability test of the starting valley at the bracket stress (e.g. a nudged-elastic-band barrier).
- No sweep, no 2× cells, no χ ≠ 0 dislocation run.

### Track 2 — ML code (`src/surrogate/`, `scripts/run_synthetic.py`, `tests/`)
- **Env "ml"** (new; the "lammps" env is untouched): python 3.11.18, numpy 2.4.6, scipy 1.17.1, pandas 3.1.0, matplotlib 3.11.2, scikit-learn 1.9.1, pytest 9.1.1, torch 2.14.1+cpu (`tools/make_ml_env.sh`, `logs/ml_env.log`).
- **Code:**
  - `MLPEnsemble`: 5 members, tanh 32×32, inputs and targets standardised; each member trains on its own bootstrap resample and adds a fixed random "prior" network (randomized prior functions, Osband et al. 2018).
  - `GPBaseline`: constant × RBF + white noise.
  - `active_learning` / `compare_strategies`: largest spread vs random, stopping rule max spread < threshold with at least 5 points.
  - `coverage`: fraction inside ±1 and ±2 spreads.
  - Inputs are (n, d) arrays: d = 1 now, d = 2 later.
- **SYNTHETIC test** (made-up curve, noise 0.01; `figures/synthetic/`, `summary.json`):

| Model, 8 points | RMSE | inside ±1 / ±2 spread (ideal 0.68 / 0.95) |
|---|---|---|
| plain ensemble (first version) | 0.025 | **0.07 / 0.12** (badly overconfident) |
| ensemble + bootstrap + random priors (used) | 0.022 | 0.63 / 0.88 |
| GP baseline | 0.010 | 0.96 / 1.00 (somewhat under-confident) |

- **Active learning** (3 starting points, 61-point pool, max 10 additions, 3 seeds): final RMSE 0.0117, 0.0178, 0.0094 (largest spread) vs 0.0160, 0.0175, 0.0163 (random). A modest gain for the spread rule. In the first version the stopping rule fired at 3 points because of the overconfident spread; it now needs at least 5 points.
- **pytest: 12/12 passed** (`logs/pytest.log`): lattice constant, C11/C12/C44 (two strain sizes), b, atom counts of both cells, tracker detects the known 2× jump, 5 ML tests.
- **Private:** `README_draft.md` (git-ignored).
- **CPU use:** ML runs used 1 thread (`nice`) next to the 7 LAMMPS ranks.

## Session 11 — Wed 28 Oct 2026 (morning): Gate B' (new bracket method), the remap bug, χ sweep

- **Plain words:**
  - Last night's χ = 0 regression failure (Gate B) was **not** the minimiser being unpredictable. It was **a bug in our step input**: the LAMMPS command that sets the box (`change_box … remap`) also stretched the crystal along y by 1 part in 10,000 on every call. That nudged every step, and near the Peierls stress the nudge was enough to push the core over the last small barrier.
  - With the bug fixed and the new, stricter minimiser settings, the χ = 0 bracket is **1.285–1.290 GPa**, identical on 7 and 8 processor counts. Every "stable" state passes a stability test.
  - So Gate B' passed, and the angle sweep was run: **χ = −30° 1.16–1.17 GPa, χ = 0 1.285–1.290, χ = +30° 2.04–2.05 GPa** (strong twinning/anti-twinning asymmetry; 2× cell within 0.9 %). Details in §5.

### 1. New bracket method (Ansh's decision) and Gate B' v1 (contaminated by the bug)
- **Settings:** CG, etol 0, ftol 1e-6 eV/Å (LAMMPS's 2-norm of the whole force vector; it implies fmax ≤ 1e-6), `min_modify dmax 0.01` Å, maxit 200000. They are options in `tools/in.step_chi`, `tools/peierls_chi.py` and the new `tools/stability_test.py`.
- **Stability test (`tools/stability_test.py`):**
  - Displace every atom randomly (uniform in ±0.001 Å per component; 2 seeds), then re-minimise at the same stress.
  - "Returned" means the DD Burgers triangle is in the same place as before (the DD core finder).
  - Timing: ~14 s per small-cell test on 7 or 8 ranks (8 ranks weren't slower this time).
- **v1 (before the fix):**
  - All Session 5 states (1.260–1.280) left the valley, so did the 2× states (1.260, 1.270), **and so did the new ramp's own last-stable states** (small 1.230, 2× 1.220).
  - The new 1.230 state left even **with no displacement at all**, while the 1.200–1.220 states and a 1.000 GPa control returned.
  - v1 brackets: small 1.230–1.235, 2× 1.220–1.225 GPa (the same on 7 and 8 ranks).
  - Kept as `sims/05_gate_bprime_v1_remapbug/`, `logs/v1remap_*`, `data/v1remap_*`, `figures/v1remap_*`.

### 2. The bug: `change_box … remap` stretches y (`sims/05_gate_bprime_v1_remapbug/diag/`)
- **Symptom:** a force-converged state (fmax 1.6×10⁻⁸ eV/Å) read back in had fmax = 6.8×10⁻³ eV/Å right after `change_box x/z/xz final <same values> remap`. Without `remap` there was no kick (fmax 1.6×10⁻⁸).
- **Each remap operation**, even with an identical box, moved **every atom along y** in proportion to its height: up to 0.0148 Å at the top surface, i.e. a uniform y-strain of ~1.0×10⁻⁴, about the shrink-wrap padding. x and z were unchanged (≤ 10⁻¹⁰ Å). Three operations in a row gave three times the kick.
- **Workarounds that did not help:** `run 0` before the remap; switching y to a fixed boundary during the remap.
- **Likely mechanism (not checked in the LAMMPS source):** the remap converts y to box fractions and back with two different y bounds of the shrink-wrapped boundary.
- **Fix:** `tools/rebox.py` writes the new box into the data file and maps the atoms affinely in x and z, leaving y untouched. It matches LAMMPS's remap in x and z to 10⁻¹⁰ Å (tested with x × 1.0005, z × 1.002, xz + 0.3 Å), and y is unchanged exactly. `in.step_chi` no longer calls `change_box`; the driver calls `rebox.py` only when the box changes (never at χ = 0).
- **What it affected:**
  - Gate B (Session 10) and Gate B' v1: yes. The Session 10 diagnosis "minimiser-path dependence, ~0.03 GPa" is **withdrawn**.
  - Session 5 (`in.step`, no `change_box`): no.
  - Gate A calibration (perfect slab, `in.block` uses remap): the start of each relaxation got the same y-strain, but a uniform y-strain relaxes away completely in a free perfect slab, so the final stresses should be unaffected. Not re-run.

### 3. Gate B' v2 (after the fix): PASS (`sims/05_gate_bprime/`, `logs/gate_bprime.out`, `data/gateBp_*.csv`)
- **Stability tests, 7 and 8 ranks, 2 seeds each:** all **returned**: Session 5 small-cell states 1.260, 1.270, 1.280 and 2× states 1.260, 1.270, plus controls (Session 5 1.260 with no displacement; 1.000 GPa). Final fmax ≤ 2.4×10⁻⁸ eV/Å (small; force tolerance). The 2× runs mostly end on "linesearch alpha is zero", fmax ≤ ~10⁻⁶ eV/Å.
- **New brackets** (ramp from the Session 5 1.200 GPa state, 10 MPa steps, then 5 MPa):

| Cell | 7 ranks | 8 ranks | Stability of the last unmoved state (2 seeds × 2 rank counts) |
|---|---|---|---|
| 146.87 Å | **1.285–1.290 GPa** | 1.285–1.290 GPa | returned (1.285) |
| 293.75 Å (2×) | **1.275–1.280 GPa** | 1.275–1.280 GPa | returned (1.275) |

- **Pass:** the brackets agree exactly between 7 and 8 ranks, and they are consistent with the stability tests (every state below the bracket, including 1.260–1.280, is a resting state).
- **χ = 0 baseline: 1.285–1.290 GPa (midpoint 1.2875)**, consistent with Session 5 (1.280–1.290) at the finer step. Reference 1.21 GPa (Chaussidon, Fivel & Rodney 2006, Section 4, read by the planning chat; corrected 28 Oct from ~1.25 GPa): +6.4 % (midpoint); 2× cell (1.2775) +5.6 %.
- **Cell size:** 1.2775 vs 1.2875 GPa = **−0.8 %** (< 3 %).
- ~~**Hypothesis check:** the planning chat's interpretation (the 1.260 jump in Gate B was an overshoot over a small remaining barrier) is confirmed in substance.~~ **Withdrawn (Ansh, 28 Oct, Session 12):** the "minimiser overshoot" hypothesis is withdrawn, like the "minimiser path-dependence" claim. The Session 5 states are true resting states; the 1.260 GPa jump in Gate B was caused by the `change_box … remap` y-stretch (a spurious kick on every step), not by the minimiser.
- **Measured bracket uncertainty:** 7 vs 8 ranks give 0 difference, so the uncertainty is the bracket width w (std w/√12; 0.0029 GPa for the 10 MPa sweep refinement). This is now `BRACKET_UNCERTAINTY_GPA` in `src/surrogate/__init__.py`, the default noise level option.

### 4. Track 2: active learning vs random, 20 seeds (SYNTHETIC; `scripts/al_seeds.py`, `figures/synthetic/al_20seeds.{json,png}`)
- **Setup:** the same synthetic curve, pool and ensemble as Session 10. The stopping rule is off, so every run goes from 3 to 13 points. For each seed both strategies start from the same 3 points. 1 thread, `nice`; paused during the 8-rank LAMMPS phase.

| Points | Spread (mean ± std) | Random (mean ± std) | Spread better | Sign test p | Paired difference (spread − random) ± s.e. | Wilcoxon p |
|---|---|---|---|---|---|---|
| 6 | 0.0439 ± 0.0211 | 0.0449 ± 0.0192 | 11/20 | 0.82 | −0.0009 ± 0.0054 | 0.90 |
| 8 | 0.0288 ± 0.0175 | 0.0332 ± 0.0137 | 12/20 | 0.50 | −0.0043 ± 0.0052 | 0.29 |
| 10 | 0.0217 ± 0.0130 | 0.0263 ± 0.0103 | 15/20 | 0.041 | −0.0046 ± 0.0041 | 0.10 |
| 13 | 0.0142 ± 0.0055 | 0.0187 ± 0.0072 | 15/20 | 0.041 | −0.0045 ± 0.0020 | 0.030 |

- **Reading:** with few points (6–8) there is no measurable difference. From 10–13 points the spread rule is better in 15 of 20 seeds, with about 24 % lower RMSE at 13 points (marginal significance, p ≈ 0.03–0.04). This holds for this synthetic 1D curve only; nothing is claimed about real data.
- **Noise option:** `noise_std` in `MLPEnsemble` (added in quadrature with `predict(include_noise=True)`) and `GPBaseline` (white-noise level fixed instead of learned). Default `BRACKET_UNCERTAINTY_GPA = 0.010/√12 = 0.0029 GPa`.

### 5. χ sweep: Peierls stress vs MRSS angle (`sims/06_chi_sweep/`, `data/peierls_chi_sweep.csv`, `figures/peierls_vs_chi.png`)
- **Plain words:**
  - Tilting the maximum-shear plane towards the **anti-twinning** side (χ > 0) makes the screw dislocation **much harder** to move: 2.04–2.05 GPa at χ = +30°, against 1.285–1.290 GPa at χ = 0.
  - Tilting it towards the **twinning** side (χ < 0) makes it **easier**: 1.16–1.17 GPa at χ = −30°.
  - Simple Schmid behaviour (only the stress resolved on the (−110) plane matters) would give a curve symmetric about χ = 0. The real curve is strongly lopsided: ratio to Schmid 0.78 at −30°, 1.38 at +30°. This is the twinning/anti-twinning asymmetry of bcc metals.
  - From −6° to +24° the core moves on (−110), the plane of the χ = 0 runs. At ±30° (and −24°) it moves on the **other {110} plane next to the MRSS plane**: (0 1 −1) at negative χ, (−1 0 1) at positive χ.
  - The 2× cell gives the same bracket at +30° and 0.9 % higher at −30°, so the small cell is converged for χ ≠ 0 too (criterion < 3 %).
- **Settings** (as Gate B'): option-2 loading with the Gate A correction (`checks/07_chi_calibration/strain_model.json`); CG, etol 0, ftol 1e-6, dmax 0.01 Å; ramp 0–2.4 GPa in 50 MPa, then 10 MPa; each step starts from the previous relaxed state; `tools/rebox.py` sets the box. 2 stability tests (±0.001 Å random displacement) at each last unmoved state.
- **"Moved" = the DD Burgers-triangle site changed** (see "Things I caught": the energy tracker missed a one-site jump at −30/−24).

| χ (°) | side | cell | bracket (GPa) | τ cos χ (mid) | Schmid 1.2875/cos χ | ratio | glide plane (DD shift) | stable at last unmoved |
|---|---|---|---|---|---|---|---|---|
| −30 | twinning | small | 1.160–1.170 | 1.009 | 1.487 | 0.784 | (0 1 −1), +60° | 2/2 |
| −30 | twinning | 2× | 1.170–1.180 | 1.018 | 1.487 | 0.790 | (0 1 −1), +60° | 2/2 |
| −24 | twinning | small | 1.180–1.190 | 1.083 | 1.409 | 0.841 | (0 1 −1), +60° | 2/2 |
| −18 | twinning | small | 1.210–1.220 | 1.156 | 1.354 | 0.898 | (−110), 0° | 2/2 |
| −12 | twinning | small | 1.220–1.230 | 1.198 | 1.316 | 0.931 | (−110), 0° | 2/2 |
| −6 | twinning | small | 1.250–1.260 | 1.248 | 1.295 | 0.969 | (−110), 0° | 2/2 |
| 0 | — | small | 1.285–1.290 (Gate B') | 1.2875 | 1.2875 | 1 | (−110) | 30/30 (Gate B') |
| +6 | anti-twinning | small | 1.390–1.400 | 1.387 | 1.295 | 1.078 | (−110), 0° | 2/2 |
| +12 | anti-twinning | small | 1.490–1.500 | 1.462 | 1.316 | 1.136 | (−110), 0° | 2/2 |
| +18 | anti-twinning | small | 1.640–1.650 | 1.564 | 1.354 | 1.215 | (−110), 0° | 2/2 |
| +24 | anti-twinning | small | 1.880–1.890 | 1.722 | 1.409 | 1.338 | (−110), 0° | 2/2 |
| +30 | anti-twinning | small | 2.040–2.050 | 1.771 | 1.487 | 1.376 | (−1 0 1), −60° | 2/2 |
| +30 | anti-twinning | 2× | 2.040–2.050 | 1.771 | 1.487 | 1.376 | (−1 0 1), −60° | 2/2 |

- **Checks:**
  - All 24 stability tests (12 brackets × 2 seeds) returned to the same site; small cell fmax ≤ 2.6×10⁻⁸ eV/Å (force tolerance), 2× cell fmax ≤ 1.6×10⁻⁷ eV/Å.
  - Every bracket lies in 0.8–3 GPa. Every last unmoved state is the starting easy core (top column 1.19–1.22 b).
  - Non-converged steps below a bracket: small cell χ = −6 (0.05, 0.60 GPa) and +6 (0.30, 0.45) stopped on "linesearch alpha is zero" with fmax ≤ 3.8×10⁻⁷ eV/Å and no move, far below the bracket. The 2× cell steps all stop that way (fmax ≤ 8.5×10⁻⁶ eV/Å), the same numerical floor as Gate B' 2×.
  - Plane geometry check: the Peach–Koehler force on the core points at −χ in the (y, z) plane (DD shift at χ = 0 is +z). At χ = ±30° the two {110} planes at 0° and ∓60° then carry the same resolved shear (cos 30°). So the ±60° moves are on planes with the full MRSS-plane Schmid factor, not a sign error.
  - First-moved core at −30/−24: one site along +60°, top column 1.54–1.55 b (core centred on one column, as in the 2× χ = 0 state of Session 7).
- **Before the "moved" fix**, χ = −30 and −24 gave 1.380–1.390 and 1.630–1.640 GPa (archived `*_trackermoved`). Those numbers are the **second** move: the core had already jumped one site on (0 1 −1) at 1.15–1.20 GPa and then rested in the new valley. By Ansh's definition the first move counts.
- **2× vs small:** +30° 0.0 %; −30° +0.9 % (midpoints 1.175 vs 1.165). Both below 3 %.
- **Status:** these results are no longer "provisional" (option 2 approved, run under Gate B' conditions). **One method change awaits Ansh's confirmation:** "moved" now comes from the DD site, not the tracker (it was the tracker in the Session 10 provisional decision).
- **Not a literature comparison:** no published χ curve for this potential has been checked yet. The qualitative trend (anti-twinning harder) is the known bcc asymmetry; a quantitative comparison is an open item.
- **Files:** `run_sweep.sh` (main), `run_rerun30.sh` (±30 after the refinement bug), `run_fix.sh` (−30/−24 with DD "moved"; 2× ±30 after the dump fix); `logs/chi_sweep.out`, `chi_rerun30.out`, `chi_fix.out`, `logs/chi_*.log`, `data/chi_*.csv`, `data/chi_sweep_stability.txt`, `logs/dd_scan_sweep.log` (DD scan of the original runs), `logs/sweep_summary.log`; DD maps `figures/ddmap_chi_*`. Archived: `*_v1finebug`, `*_trackermoved`, `*_dumpbug`.
- **Commands:** `python tools/dd_scan.py sims/06_chi_sweep/chi_small_*`; `python tools/sweep_summary.py 1.2875 --stab data/chi_sweep_stability.txt`.


## Session 12 — Wed 28 Oct 2026 (Block 1): loading re-check, χ = −24 core, surrogate on the real data, live active learning

- **Plain words:**
  - The box-strain loading was re-checked through the corrected path (`rebox.py`): it is unchanged to 0.00007 GPa, so Gate A stands.
  - At χ = −24° the core's first move is one site sideways onto the (0 1 −1) plane, even though that plane gets 11 % less shear than (−110). The moved core is spread around one atom column.
  - The surrogate learns the χ curve much better when it learns the *deviation from Schmid's law* rather than τ_P itself. But all models are over-confident on this data: the curve has wiggles of ~0.02–0.03 GPa that the 0.0029 GPa noise level does not allow for.
  - Live active learning ran for 3 rounds; results in §5.
- **Decisions recorded** (see "Decisions so far"): "moved" from the DD site is official; the "minimiser overshoot" and "minimiser path-dependence" explanations are withdrawn; the χ-curve literature status.

### 1. Gate A spot re-check through the rebox path (`checks/08_gateA_rebox/`): PASS
- **What:** the perfect slab of Gate A (2400 atoms, `boundary p s p`) built once (`in.build`, no strain, `slab0.data`). Each strain is written into the data file by `tools/rebox.py` (x, z affine, y untouched), then read back, loaded with the surface forces and relaxed (`in.measure`: same groups, interior average and minimiser as `checks/04_pure_shear/in.block`, but no `change_box`). The unloaded reference is measured the same way and subtracted.
- **Cases:** χ = −30, −12, +12, +30 at τ = 1.5 and 2.0 GPa, with (a) the calibrated strains of `calibration.csv` (the Gate A points) and (b) the τ-polynomial strains of `strain_model.json` (what the sweep driver actually applied).
- **Criteria (Gate A):** σ_xy and σ_xz within 1 % of target, all other components < 2 % of τ.

| Strains | Points | max σ_xy error | max σ_xz error | max other (% of τ) | max change vs the `change_box` measurement | Result |
|---|---|---|---|---|---|---|
| calibrated | 8 | 0.002 % | 0.053 % | 0.081 % | 0.000071 GPa | **PASS 8/8** |
| polynomial model (used by the sweep) | 8 | 0.002 % | 0.266 % (χ = +12, τ = 1.5) | 0.048 % | 0.0011 GPa | **PASS 8/8** |

- The slab height ly = 38.368712 Å is identical in every run (y untouched), as intended.
- **Conclusion:** the `change_box … remap` y-stretch did not affect the Gate A calibration (as expected for a free perfect slab, now verified). The sweep's loading stands.
- **Files:** `checks/08_gateA_rebox/in.build`, `in.measure`, `recheck.py`, `recheck.csv`, `log.*`; `logs/gateA_rebox.out`. **Command:** `cd checks/08_gateA_rebox && ../../tools/launch_safe.sh gateA_rebox python recheck.py`

### 2. χ = −24°: what the core does at the bracket (`figures/ddmap_chi_small_m24_last_unmoved.png`, `_first_moved.png`)
- **Last unmoved (1.180 GPa):** the compact easy core at its starting site. Burgers triangle 0.347 / 0.315 / 0.338 b (skewed a little by the stress), top column 1.200 b (unloaded core 1.10 b). Two stability tests returned (Session 11).
- **First moved (1.190 GPa, converged, 638 CG iterations):** the Burgers triangle moved by (dy, dz) = (+2.019, +1.166) Å: one site (2.331 Å) at **+60° from the glide direction**, i.e. **on the (0 1 −1) plane**, not on (−110).
- **Resolved shear on the plane it used vs (−110):** with the MRSS plane at χ = −24°, the (0 1 −1) plane is 36° away and (−110) is 24° away. So at 1.190 GPa the resolved shear is τ cos 36° = **0.963 GPa** on (0 1 −1) against τ cos 24° = **1.087 GPa** on (−110) (0.81 vs 0.91 of τ). **The core moved on the plane with 11 % less resolved shear**: a non-Schmid effect (the twinning-side {110} plane is easier).
- **Core shape changed:** at 1.190 GPa the core is no longer a compact triangle. It is spread around **one atom column**, at (dy, dz) = (+0.67, +1.17) Å, whose bonds carry 0.474 and 0.470 b (sum 1.552 b). This is the same column-centred shape as the 2× cell χ = 0 state at 1.280 GPa (Session 7, 1.58 b) and the χ = −30 first-moved state (1.54 b), i.e. the split-like core known as a Mendelev-potential artifact (Gilbert et al. 2013, read by the planning chat). The column lies between the old and the new triangle.
- **Command:** `cd sims/06_chi_sweep/chi_small_m24 && SCREW_NZ=21 python ../../../tools/ddmap_data.py steps/fine_1.190.data <png>` (same for `fine_1.180`).

### 3. Surrogate on the real data: dataset and leave-one-out (`scripts/surrogate_loo.py`)
- **Dataset** `data/surrogate_train_v1.csv`: the 10 small-cell sweep angles (bracket midpoints, `data/peierls_chi_sweep.csv`) plus the χ = 0 baseline 1.2875 GPa (Gate B'); 11 points, all in the 146.87 Å cell. The 2× rows (±30) are left out (different cell). Noise σ = 0.0029 GPa for every point. The χ = 0 Gate B' bracket was re-checked with the DD scan (`tools/dd_scan.py sims/05_gate_bprime/ramp_small_np7`): 1.285–1.290 GPa, as before.
- **Models** (settings fixed in Sessions 10–11, nothing tuned here): `gp_tau` (GP, constant × RBF, noise fixed to σ); `mlp_tau` (5-member MLP ensemble, bootstrap + randomized priors, σ added in quadrature); `gp_ratio` / `mlp_ratio`: the same, trained on r = τ_P cos χ / τ_P(0) (`src/surrogate/physics.py`) and converted back.
- **Check that the ratio's τ_P(0) leaks nothing:** it is only a scale. The ratio models give identical τ_P predictions (to 10⁻⁶) with τ_P(0) = 1.2875 or 1.0, also when χ = 0 is left out.

| Model | LOO RMSE (GPa) | RMSE, 9 interior points | max \|err\| | inside ±1 / ±2 spread (ideal 0.68 / 0.95) |
|---|---|---|---|---|
| gp_tau | 0.0460 | 0.0276 | 0.121 (χ = +30) | 0.18 / 0.55 |
| mlp_tau | 0.0658 | 0.0475 | 0.165 (χ = +30) | 0.55 / 0.82 |
| **gp_ratio (best)** | **0.0291** | **0.0219** | 0.066 (χ = +30) | 0.45 / 0.64 |
| mlp_ratio | 0.0326 | 0.0348 | 0.077 (χ = +24) | 0.64 / 0.73 |

- **Best model: gp_ratio** (lowest LOO RMSE, the rule fixed in the script before it ran). Full per-angle table: `data/surrogate_loo.csv`.
- **The physics-informed form helps:** RMSE −37 % for the GP and −50 % for the MLP, and the extrapolation to χ = +30 improves most (GP error −0.121 → −0.066 GPa). Reason: Schmid's 1/cos χ is built in, so the model only has to learn the smoother asymmetry r(χ).
- **All four models are over-confident** (±2 coverage 0.55–0.82, ideal 0.95). The GP errors alternate in sign at ~±0.025 GPa from point to point: the measured curve has wiggles of that size, about 10 × the 0.0029 GPa bracket noise. The fixed small noise forces the GP to pass through every point, so its spread between points is too small. **Not fixed** (that would be tuning on this data); options for Ansh are in "Next steps".
- **χ = 0 stands out:** every model over-predicts it by 0.025–0.029 GPa (|err|/spread 3.4–5.3). In the data, the slope of r is 0.0051 per degree on −6…0 but 0.0129 per degree on 0…+6, a kink at χ = 0. The χ = 0 point is also the only one from a different protocol (Gate B': ramp from the 1.20 GPa state in 10/5 MPa steps; the sweep ramps from 0 in 50/10 MPa steps). Not resolved: a re-run of χ = 0 with the sweep protocol would tell (Next steps).
- ML runs: env `ml`, 1 thread, `nice`.

### 4. Retrospective active learning on the 11-point pool (`scripts/al_retro.py`, `data/al_retro_*.json`)
- **Setup:** the oracle returns the measured value (no new runs). 3 random starting points (the same for both strategies), then add the point with the largest spread or a random one, until all 11 are in. RMSE on all 11 after each refit, 20 seeds.

| Points | gp_ratio: spread | gp_ratio: random | spread better (sign p) | gp_tau: spread | gp_tau: random | spread better (sign p) |
|---|---|---|---|---|---|---|
| 4 | 0.092 ± 0.095 | 0.146 ± 0.115 | 15/18 (0.008) | 0.125 ± 0.100 | 0.176 ± 0.110 | 12/18 (0.24) |
| 5 | 0.064 ± 0.067 | 0.128 ± 0.098 | 18/20 (<0.001) | 0.098 ± 0.102 | 0.140 ± 0.094 | 15/20 (0.04) |
| 6 | 0.053 ± 0.054 | 0.087 ± 0.072 | 15/20 (0.04) | 0.076 ± 0.096 | 0.098 ± 0.083 | 15/20 (0.04) |
| 7 | 0.036 ± 0.032 | 0.084 ± 0.076 | 17/20 (0.003) | 0.060 ± 0.074 | 0.077 ± 0.074 | 10/20 (1.0) |
| 8 | 0.023 ± 0.018 | 0.063 ± 0.056 | 19/20 (<0.001) | 0.042 ± 0.052 | 0.063 ± 0.062 | 17/20 (0.003) |
| 9 | 0.019 ± 0.016 | 0.042 ± 0.059 | 13/20 (0.26) | 0.036 ± 0.048 | 0.043 ± 0.053 | 13/19 (0.17) |
| 10 | 0.010 ± 0.004 | 0.019 ± 0.025 | 9/18 (1.0) | 0.022 ± 0.035 | 0.025 ± 0.025 | 11/18 (0.48) |

(RMSE in GPa, mean ± std over 20 seeds; at n = 3 and 11 both strategies are identical by construction.)
- **Reading:** with gp_ratio the spread rule roughly halves the RMSE at 5–8 points (better in 15–19 of 20 seeds). The advantage fades at 9–10 points, where few candidates are left.
- **Honest caveat:** most of the gain comes from the spread rule taking the **two end points (±30°) first** (in 17 of 20 seeds its first picks were exactly the missing end points). Random choice had both ends in by 5 points in only 1 of 20 seeds. Without the ends the GP must extrapolate, which costs the most. So on this pool the spread rule mostly rediscovers "measure the ends of the range first". A simple end-first rule was not tested. The MLP ensemble was not run here (≈30 min single-core).

### 5. Live active learning (`sims/07_al_live/`, `data/al_live_*.csv`): 2 rounds run, round 3 stopped
- **Plain words:** the surrogate chose the next angle itself and, **before** each LAMMPS run, wrote down what it expected (`data/al_live_predictions.csv`, time-stamped). Round 1 came out inside its predicted band. Round 2 came out far above the band, and the core did something new there: it changed shape in place instead of jumping. That raises a definition question for Ansh. Round 3 was stopped (below), so the claimed loop is shown for two rounds, with one success and one honest miss.
- **Setup per round** (`run_al_live.sh`): best LOO model (gp_ratio) refitted on all data; pool χ = −30…+30 in 1° steps minus angles already run; pick = largest spread; pre-register; Gate A calibration of the new angle through the rebox path (`checks/08_gateA_rebox/calibrate_chi.py`, same criteria, added to `strain_model_al.json`); small-cell ramp exactly as the sweep (option 2, etol 0, ftol 1e-6, dmax 0.01, 0–2.4 GPa in 50 MPa, 10 MPa refinement, DD-site "moved"); 2 stability tests at the last unmoved step. Run estimate 25–40 min; rounds 1–2 took 20 min.

| Round | χ | Pre-registered prediction (±1 spread) | ±2 band | Measured bracket (DD site) | z | Inside ±2? | Plane / what moved | Stability (last unmoved) | Gate A of the angle |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −28° | 1.1581 ± 0.0067 | 1.1446–1.1715 | **1.160–1.170** (mid 1.165) | +1.04 | **yes** | one jump (2.331 Å) at +60° onto (0 1 −1), first-moved core column-centred (1.545 b), as at −30/−24 | 2/2 returned (fmax ≤ 1.7e−8) | PASS 8/8 |
| 2 | +28° | 2.0086 ± 0.0094 | 1.9899–2.0274 | **2.110–2.120** (mid 2.115) | +11.3 | **no** | **no jump:** Burgers triangle shifted 1.346 Å along +y (down → up triangle); core became split-like around a corner column of the starting triangle (1.803 b) | 2/2 returned (fmax ≤ 1.7e−8) | PASS 8/8 |
| 3 | +3° (pre-registered 1.3438 ± 0.0389) | — | — | **not run** | — | — | — | — | **FAIL 7/8** |

- **Round 2 in detail:** at 2.110 GPa the core is the compact easy core at its start (top column 1.225 b). At 2.120 GPa (converged, 763 iterations) the Burgers triangle sits on the neighbouring "up" triangle that shares an edge with the start triangle (1.346 Å along y, 0 along z). The core is spread around column (+0.67, +1.17) Å, one corner of the original triangle, with bonds 0.491 / 0.461 / 0.420 b. So **the core changed shape in place; it did not glide one jump.** The DD-site rule counts this as "moved" (site changed > 0.01 Å). Ansh's definition ("leaves the starting valley by at least one jump") does not. **This first case where the two disagree is for Ansh to decide.** Until then the +28° value is flagged. The first true jump at +28° (somewhere above 2.120 GPa) was not measured. All earlier sweep angles moved by ≥ one full jump (checked in `data/peierls_chi_sweep.csv`: shifts 2.331 or 6.994 Å).
- **Round 3:** the pick (+3°, pre-registered 1.3438 ± 0.0389 GPa) was made on a training set holding the recorder bug's wrong values (below). Its Gate A calibration then failed one of 8 points: at τ = 0.5 GPa the τ-polynomial gave σ_xz 1.23 % off target (target 0.026 GPa, i.e. 0.0003 GPa absolute); the 4 calibrated points and the other 3 checks pass. The failure is about the relative criterion on a very small σ_xz at small χ. Relaxing it is Ansh's call. The ramp did not start (the strain model had no +3 entry, so `peierls_chi.py` stopped with a KeyError; no LAMMPS step ran). **Round 3 is void**: its row stays in `al_live_predictions.csv` (never rewritten) and is not drawn in the figure.
- **Did the max spread shrink? No.** Max pool spread before round 1: 0.0067 GPa. After round 1 (corrected data): 0.0066 (−2 %). After round 2: **0.0305 GPa (4.5× larger)**. The +28° point (r = 1.45, above the +24/+30 neighbours) cut the GP's length scale from 9.1° to 5.2°, so the model became less certain everywhere between points. That is the honest response to a surprising point, but it means 2 rounds did not tighten the model.
- **Recorder bug and its effect (post hoc, labelled counterfactual):** `al_live_record.py` first took the first "moved" row in file order (the coarse step), so rounds 1 and 2 were recorded as 1.160–1.200 and 2.110–2.150 GPa, and the round-2 and round-3 picks were fitted on mids 1.180 and 2.130 instead of 1.165 and 2.115. Refitting with the correct value: round 2 would have picked +27° (spread 0.0066) instead of +28°, and the prediction at +28° would have been 2.0095 ± 0.0065 instead of 2.0086 ± 0.0094, so the miss at +28° is not caused by the bug. Round 3 with correct data would still pick ±3° (spread 0.0305). Fixed; the buggy file is archived as `data/al_live_results_recorderbug.csv`.
- **Figure** `figures/surrogate_chi_v1.png`: sweep points, gp_ratio mean ± 2 spread fitted on all 13 points (v1 + rounds 1–2), Schmid curve, pre-registered predictions with ±2 bars, AL picks numbered (2* = flagged), twinning side χ < 0 and anti-twinning side χ > 0; bottom panel the same as r(χ).
- **Files:** `scripts/surrogate_common.py`, `surrogate_loo.py`, `al_retro.py`, `al_live_pick.py`, `al_live_record.py`, `plot_surrogate.py`; `src/surrogate/physics.py`; `checks/08_gateA_rebox/calibrate_chi.py`, `calibration_al.csv`, `strain_model_al.json`; `sims/07_al_live/` (run folders for −28, +28; +3 empty); `data/al_live_predictions.csv`, `al_live_results.csv`, `al_live_pool_round*.csv`, `al_live_stability.txt`, `chi_small_m28.csv`, `chi_small_p28.csv`; `logs/al_live.log`, `.out`, `logs/chi_small_m28.log`, `chi_small_p28.log`; DD maps `figures/ddmap_chi_small_m28_*`, `ddmap_chi_small_p28_*`.
- **Commands:** `cd sims/07_al_live && ../../tools/launch_safe.sh al_live ./run_al_live.sh`; `cd scripts && python surrogate_loo.py; python al_retro.py gp_ratio; python plot_surrogate.py` (ml env, 1 thread).

## Session 13 — Wed 28 Oct 2026 (Block 2): decisions D1–D4, literature correction, audit, Vitek benchmark, +28 and χ = 0 runs

- **Plain words:**
  - **Reference corrected:** the published Peierls stress for our potential at χ = 0 is **1.21 GPa** (Chaussidon et al. 2006 itself), not the ~1.25 GPa we had taken from a later paper's restatement. Ours is 6–9 % above it.
  - **Our χ curve has the same shape as Chaussidon's but a stronger asymmetry:** fitted with the same two-parameter formula, a = 0.83 ± 0.04 for us against 0.61 for them. Our values are 1–6 % above theirs on the twinning side and up to 15 % above on the anti-twinning side.
  - **The χ = 0 "kink" is explained:** run with the same loading recipe as the other angles, χ = 0 gives **1.310–1.320 GPa**, 2.1 % above the Gate B' value. The Gate B' loading lacked the small second-order correction, which tilts the effective shear plane by −1.7°. The new value lies right on the curve through its neighbours (fit: 1.3165).
  - **χ = +28 is still not settled, and this time it is the rule, not the run.** Under D1 the +28 ramp "moved" at 2.170 GPa, but the atoms did not move then (≤ 0.013 Å). Only the DD finder's choice of triangle flipped. The real event happened at 2.120 GPa: the core turned into a split-like core on one column. That is **the same event that the twinning-side angles (−30, −28, −24) count as their first move.** The stability test at 2.160 GPa failed for the same reason (labelling near-tie). **Stopped before live round 3** (B4 not run) for Ansh's decision.
- **Decisions recorded** (see "Decisions so far"): D1–D4 and the literature correction.

### 1. Literature correction: 1.25 → 1.21 GPa
- Every mention of the old reference (~1.25 GPa, as restated by Gilbert et al. 2013) was found by grep and replaced with **1.21 GPa (Chaussidon, Fivel & Rodney, *Acta Mater.* 54, 3407 (2006), Mendelev potential 2, free boundaries, 1210 MPa at χ = 0, text Section 4; read by the planning chat)**. `PROMPTS.md` keeps its verbatim history.
  - `NOTES.md`: Session 4 "Things I caught" (PAD shear entry), Session 5 result table, Session 6 cell-size table, Session 11 §3 baseline line, "Decisions so far" (project potential), "Next steps" item 9.
  - private notes (not in the repository): story-so-far line, checks table (2 rows), PAD story, one quoted line, numbers list, "why Mendelev", Mendelev-vs-DFT answer.
  - `README_draft.md` (private): validation item 6.
- **New percentages** (ours vs 1.21 GPa): Session 5 bracket 1.280–1.290 → +5.8 % to +6.6 % (mid +6.2 %); Gate B' 1.2875 → +6.4 %; 2× 1.2775 → +5.6 %; B2 (sweep protocol, below) 1.315 → **+8.7 %**.

### 2. A1 — audit of every first move (`scripts/audit_moves.py`, `data/audit_moves.csv`)
- Each first-moved state file re-analysed with the DD core finder. "Column-centred" = top column ≥ 1.4 b (compact easy core under load: 1.10–1.23 b).

| Run | χ | Bracket (GPa) | DD shift (Å) | Jumps | Angle | Orientation | Plane | Top column (b) | Core at first move |
|---|---|---|---|---|---|---|---|---|---|
| small | −30 | 1.160–1.170 | 2.331 | 1 | +60° | down | (0 1 −1) | 1.540 | column-centred |
| small | −24 | 1.180–1.190 | 2.331 | 1 | +60° | down | (0 1 −1) | 1.552 | column-centred |
| small | −18 | 1.210–1.220 | 6.994 | 3 | 0° | down | (−110) | 1.182 | compact |
| small | −12 | 1.220–1.230 | 6.994 | 3 | 0° | down | (−110) | 1.135 | compact |
| small | −6 | 1.250–1.260 | 2.331 | 1 | 0° | down | (−110) | 1.208 | compact |
| small | 0 (Gate B') | 1.285–1.290 | 6.994 | 3 | 0° | down | (−110) | 1.160 | compact |
| small | +6 | 1.390–1.400 | 6.994 | 3 | 0° | down | (−110) | 1.146 | compact |
| small | +12 | 1.490–1.500 | 2.331 | 1 | 0° | down | (−110) | 1.215 | compact |
| small | +18 | 1.640–1.650 | 2.331 | 1 | 0° | down | (−110) | 1.218 | compact |
| small | +24 | 1.880–1.890 | 6.994 | 3 | 0° | down | (−110) | 1.134 | compact |
| small | +30 | 2.040–2.050 | 6.994 | 3 | −60° | down | (−1 0 1) | 1.159 | compact |
| 2× | −30 | 1.170–1.180 | 2.331 | 1 | +60° | down | (0 1 −1) | 1.539 | column-centred |
| 2× | +30 | 2.040–2.050 | 2.331 | 1 | −60° | down | (−1 0 1) | 1.213 | compact |
| live 1 | −28 | 1.160–1.170 | 2.331 | 1 | +60° | down | (0 1 −1) | 1.545 | column-centred |
| live 2 | +28 | 2.110–2.120 | **1.346** | 0.58 | +90° | **up** | none | 1.803 | column-centred — **FLAG** |

- **χ = +30 is a true jump** (3 jumps on (−1 0 1) in the small cell, 1 in the 2× cell, compact core at the new site). B3 is not needed.
- **Only +28 is flagged** under the shift test. But note (found in B1, §6): every twinning-side first move (−30, −28, −24, 2× −30) is also a column-centred state, and at +28 the same kind of state is labelled differently.

### 3. A2 — D1, D2, D4 in the code; tests
- **D1** (`tools/moved_rule.py`, used by `tools/peierls_chi.py --moved dd` and `scripts/al_live_record.py`): moved = DD site shift ≥ 2.331 − 0.01 Å onto a 'down' (easy-core-equivalent) triangle; other shifts > 0.01 Å = core-change event (logged, column `core_change`, ramp continues). The old rule is kept as `--moved dd_s12` to reproduce Session 11–12 runs. New driver options for continuing a ramp: `--ref`, `--dd0`, `--start_core`. `al_live_record.py` now re-classifies every step (orientation from the step file where the csv predates it), merges a ramp and its continuation, and replaces the row of a re-recorded round. `stability_test.py` now applies `--model` also at χ = 0 (Gate B' passed no `--model`, so its results are unchanged).
- **D2** (`tools/gate_a.py`, used by `checks/08_gateA_rebox/calibrate_chi.py`): σ_xz passes if within 1 % of target **or** |Δσ_xz|/τ ≤ 0.1° (0.001745 rad). **Changed after the χ = +3 failure**, because 1 % of a near-zero target (τ sin 3°) is an extreme demand: the failing point was 0.00032 GPa off at τ = 0.5 GPa, a loading-plane tilt of 0.037°. It is an OR, so no earlier pass can become a fail.
- **D4** (`src/surrogate/gp.py`: `noise_bounds`; `scripts/surrogate_common.py`: `gp_tau_d4`, `gp_ratio_d4`): GP white noise learned within [0.0029, 0.1] GPa (r units: divided by τ0). MLP unchanged. Live-AL training uses only rows recorded under D1.
- **Tests** (`tests/test_session13.py`): (i) DD shift 1.346 Å ('up') is a core change, 2.331 Å along z and along +60° is moved; (ii) Δσ_xz = 0.0003 GPa at τ = 0.5, χ = 3 passes D2, 0.01 GPa fails, a large-target pass is unchanged; (iii) learned noise stays within [0.0029, 0.1] GPa on smooth and noisy data; (iv) Vitek formula with s0 = 1604 MPa, a = 0.61 gives 1229 ± 1 MPa at χ = 0. **Full suite: 16/16 pass** (`python -m pytest tests`, ml env).

### 4. A3 — χ = +3 calibration re-checked with D2 (stored points, `checks/08_gateA_rebox/recheck_d2.py`)

| τ (GPa) | kind | Δσ_xz (GPa) | % of target | tilt (°) | old rule | D2 |
|---|---|---|---|---|---|---|
| 1.0 | calibrated | +0.00003 | 0.048 | 0.0014 | PASS | PASS |
| 1.5 | calibrated | −0.00002 | 0.023 | 0.0007 | PASS | PASS |
| 2.0 | calibrated | −0.00008 | 0.081 | 0.0024 | PASS | PASS |
| 2.4 | calibrated | −0.00000 | 0.001 | 0.0000 | PASS | PASS |
| 0.5 | model check | −0.00032 | 1.227 | 0.0368 | **FAIL** | **PASS** |
| 1.25 | model check | +0.00010 | 0.145 | 0.0044 | PASS | PASS |
| 1.75 | model check | +0.00024 | 0.266 | 0.0080 | PASS | PASS |
| 2.2 | model check | −0.00101 | 0.876 | 0.0263 | PASS | PASS |

- **χ = +3: 7/8 → 8/8 under D2.** σ_xy and the other components were far inside their limits throughout. `calibration_al.csv` now carries `err_xz_angle_deg`, `rule` and `gate_A_D2` for every point (−28, +28, +3). The +3 strain model was not saved (the old run failed); a new calibration would be run before any +3 ramp.

### 5. A4 — Vitek-type benchmark against Chaussidon et al. 2006 (`scripts/fit_vitek.py`, `data/vitek_fit.json`, `data/vitek_compare.csv`)
- **Model:** τ(χ) = s0 / (cos χ + a cos(60° + χ)), least squares on our 8 points that glide on (−110): χ = −18, −12, −6, 0 (Gate B'), +6, +12, +18, +24. Not used: −30, −28, −24 ((0 1 −1)), +30 ((−1 0 1)), +28 (pending).
- **Assumption (not checked against the paper):** Chaussidon's χ has our sign convention (χ > 0 = anti-twinning).
- **Fit: s0 = 1862 ± 23 MPa, a = 0.829 ± 0.041** (RMS residual 18 MPa; τ(0) from the fit 1316.5 MPa). **Chaussidon (planning chat): s0 = 1604 MPa, a = 0.61** (τ(0) from their fit 1229 MPa; measured 1210 MPa).
- **Sanity test:** their fit gives 1229.1 MPa at χ = 0 (pytest, ±1 MPa): pass.

| χ | plane | ours (GPa) | our fit | residual | Chaussidon fit | ours / theirs |
|---|---|---|---|---|---|---|
| −30 | (0 1 −1) | 1.165 | 1.176 | −0.011 | 1.150 | 1.013 |
| −28 | (0 1 −1) | 1.165 | 1.174 | −0.009 | 1.146 | 1.017 |
| −24 | (0 1 −1) | 1.185 | 1.175 | +0.010 | 1.140 | 1.039 |
| −18 | (−110) | 1.215 | 1.188 | +0.027 | 1.142 | 1.064 |
| −12 | (−110) | 1.225 | 1.215 | +0.010 | 1.157 | 1.059 |
| −6 | (−110) | 1.255 | 1.257 | −0.002 | 1.186 | 1.059 |
| 0 (Gate B') | (−110) | 1.2875 | 1.317 | −0.029 | 1.229 | 1.047 |
| +6 | (−110) | 1.395 | 1.398 | −0.003 | 1.291 | 1.081 |
| +12 | (−110) | 1.495 | 1.509 | −0.014 | 1.375 | 1.087 |
| +18 | (−110) | 1.645 | 1.658 | −0.013 | 1.488 | 1.105 |
| +24 | (−110) | 1.885 | 1.862 | +0.023 | 1.641 | 1.149 |
| +30 | (−1 0 1) | 2.045 | 2.151 | −0.106 | 1.852 | 1.104 |

- **Reading:** same direction and functional shape. Our curve lies 1–6 % above Chaussidon's fit on the twinning side and 8–15 % above on the anti-twinning side, so **our asymmetry (a = 0.83) is stronger than theirs (0.61)**, well beyond the fit's standard error. Possible causes (not tested): cell geometry and boundary conditions (their free-boundary cell vs our PAD slab), the loading method, the "moved" criterion. At +30 the Vitek formula (for (−110) glide) over-predicts our value by 0.11 GPa, as expected: the core moves on (−1 0 1) there.
- **README draft:** "magnitude not benchmarked" replaced by "benchmarked against Chaussidon et al. 2006 (same potential): see the Vitek-fit comparison".

### 6. A5 — leave-one-out with D4 (`scripts/surrogate_loo_d4.py`, `data/surrogate_loo_d4*.{csv,json}`)
- Point sets: **all** = v1 (11) + live round 1 (−28) = 12 points (+28 pending, not used); **(−110) only** = 8 points (diagnostic). Rule fixed before running: the D4 model with the lower RMSE on "all" is the live-AL model.

| Set | Model | LOO RMSE (GPa) | ±1 / ±2 coverage | learned noise (GPa, full set) |
|---|---|---|---|---|
| all | gp_tau_d4 | **0.0316** | 0.58 / 0.67 | 0.0158 |
| all | gp_ratio_d4 | 0.0388 | 0.58 / 0.67 | 0.0156 |
| all | gp_ratio (fixed 0.0029, S12) | 0.0277 | 0.25 / 0.67 | 0.0029 (fixed) |
| (−110) only | gp_tau_d4 | 0.0437 | 0.50 / 0.75 | 0.0165 |
| (−110) only | gp_ratio_d4 | 0.0342 | 0.50 / 0.75 | 0.0142 |
| (−110) only | gp_ratio (fixed) | 0.0538 | 0.38 / 0.50 | 0.0029 (fixed) |

- **Model for the rest of live AL: gp_tau_d4.**
- **Reading:** the learned noise settles at ~0.016 GPa, about 5× the bracket noise: the data's point-to-point structure. ±1 coverage improves (0.25 → 0.58) but ±2 stays at 0.67: −6 and +24 are still 4+ spreads off. On all points the fixed-noise ratio GP still has the lowest RMSE; on the (−110) subset the D4 ratio GP is best. With D4 the ratio form no longer helps on "all", because the +30 extrapolation (a different glide plane) is worse.

### 7. A6 — retrospective AL with D4: spread vs random vs "ends first, then bisect" (`scripts/al_retro_d4.py`, `data/al_retro_d4.json`)
- Model gp_tau_d4; pool = the 12 points of A5; 20 seeds. Bisect rule (deterministic): −30, 0, +30, then the pool point nearest the midpoint of the largest gap (order −18, +12, −12, +18, −24, −6). Also the spread rule started from −30, 0, +30.

| n | spread (20 seeds) | random (20 seeds) | bisect | spread from −30/0/+30 | spread better than random | spread better than bisect |
|---|---|---|---|---|---|---|
| 5 | 0.0223 ± 0.0118 | 0.1037 ± 0.0943 | 0.0143 | 0.0174 | 15/20 | 1/20 |
| 6 | 0.0191 ± 0.0055 | 0.0497 ± 0.0597 | 0.0145 | 0.0177 | 13/20 | 1/20 |
| 7 | 0.0165 ± 0.0018 | 0.0295 ± 0.0239 | 0.0164 | 0.0284 | 14/20 | 12/20 |
| 8 | 0.0147 ± 0.0010 | 0.0205 ± 0.0137 | 0.0158 | 0.0134 | 15/20 | 15/20 |
| 9 | 0.0136 ± 0.0009 | 0.0181 ± 0.0101 | 0.0157 | 0.0131 | 13/20 | 18/20 |

- **Reading:** at 5–6 points the simple bisect rule is clearly better than the spread rule (spread wins 1/20): the spread rule's early advantage over random is mostly "get the ends". From 8–9 points the spread rule beats bisect in 15–18 of 20 seeds, by ~0.001–0.002 GPa, which is small next to the ~0.016 GPa learned noise.

### 8. B1 — χ = +28: continuation from 2.120 GPa (`sims/07_al_live/chi_small_p28_cont/`, `data/chi_small_p28_cont.csv`): gate FAIL
- **Run:** from the 2.120 GPa state, 10 MPa steps from 2.13 GPa, D1 rule, start site = the inserted easy site (`--dd0 0,0`), same settings as the sweep; 5 steps, 56 s.
- **Result as the code reports it:** core-change events at 2.13–2.16 GPa (site (+1.346, 0) Å, 'up'); "moved" at **2.170 GPa** (site (+2.019, +1.166) Å, 'down', one jump at +60°). Core-change bracket 2.110–2.120 GPa; D1 jump bracket 2.160–2.170 GPa.
- **But the 2.170 "move" is not a physical event:**
  - max non-affine atom displacement 2.160 → 2.170: **0.013 Å**, identical to an ordinary step (2.150 → 2.160: 0.013 Å). Compare 2.110 → 2.120 (the core change): **0.548 Å**, and χ = −28 at its first move (1.160 → 1.170): 0.442 Å;
  - the core is the same before and after: centred on column C at (+0.67, +1.17) Å, 1.804 b;
  - only the Burgers-triangle label changed: the third bond of the 'up' triangle (0.049 b) vs the +60° 'down' triangle (0.254 b) is a near-tie around column C, and a 10 MPa step tipped the finder's choice.
- **Stability test at the last unmoved step (2.160 GPa): 0/2 returned.** Both relaxed to the same column-centred core (1.804 b) with the 'down' label (fmax ≤ 1.6e−8). **Gate fails**, for the same labelling reason.
- **What this shows about D1:** the event at +28 / 2.120 GPa (compact → split-like core on column C, a 0.5 Å rearrangement) is the **same kind of event** as the first moves at −30, −28, −24 (and 2× −30), which A1 shows are also column-centred on C with the 'down' +60° label. D1 counts those as moves and this one as a core change, because of which neighbouring triangle the finder names. **Decision needed (Ansh):** either (a) the split-like column-C state counts as "moved" (then +28: 2.110–2.120 GPa, and the twinning side unchanged), or (b) it doesn't, and "moved" needs the core to arrive compact at an equivalent easy site (then −30, −28, −24 and +28 all need their ramps continued to the next compact state). Not decided by Claude.
- `al_live_results.csv` round 2 is left as the Session 12 row (not rule D1, not used in fits); the pre-registered prediction is untouched.

### 9. B2 — χ = 0 with the sweep protocol (D3) (`sims/08_block2/chi_small_0/`, `data/chi_small_0.csv`): done
- **Run:** `peierls_chi.py --chi 0` with the calibrated χ = 0 correction (`strain_model_al.json` "0", from checks/07: small ε_xx, ε_zz, γ_xz and surface-stress correction, e.g. at τ = 1: ε_xx −7.4e−5, ε_zz +1.06e−4, γ_xz +3.8e−4), 0–2.4 GPa in 50 MPa, 10 MPa refinement, D1; 33 steps, 8 min.
- **Result: 1.310–1.320 GPa** (mid 1.315), first move 3 jumps along (−110), compact core, no core-change events. Stability at 1.310: **2/2 returned** (fmax ≤ 1.8e−8). Non-converged steps below the bracket: 1.15 and 1.31 GPa stopped on "linesearch alpha is zero" with fmax 1.6e−7 / 9.5e−8 eV/Å and no move (numerical floor, as before).
- **Compared with Gate B' 1.285–1.290: +2.1 %** (midpoints). The difference is the loading: Gate B' (and Sessions 5–6) used the uncorrected surface-force loading, which carries a second-order σ_xz = −0.038 GPa at 1.285 GPa (Session 9), a −1.7° tilt of the MRSS plane towards the twinning side. With the fitted slope dτ/dχ = 0.0117 GPa/° at χ = 0, that tilt alone predicts −0.020 GPa against the −0.0275 GPa observed (σ_xx, σ_zz leftovers not included; inference, not a separate test). The B2 value sits on the Vitek fit through the neighbouring angles (τ(0) = 1.3165 GPa): **the χ = 0 kink of Session 12 was a loading-protocol difference.**
- **Against the reference:** 1.315 vs 1.21 GPa (Chaussidon 2006) = +8.7 %. The cell-size test (2× cell) was done with the uncorrected loading; not repeated.
- **Not decided:** which χ = 0 value is the baseline from now on (Ansh). The surrogate training set and the Schmid curves still use 1.2875 until then.

### 10. B3, B4
- **B3** not needed: +30 is a true jump (A1).
- **B4 not run:** the B1 gate failed, and the training data for round 3 depends on the D1 question (+28 and possibly −30/−28/−24) and on the χ = 0 baseline. The round-3 row of Session 12 is marked "voided: recorder bug, superseded" in `data/al_live_predictions.csv` (new `status` column).

### 11. Figures
- `figures/peierls_vs_chi.png` (new script `scripts/plot_peierls_vs_chi.py`, replaces the `tools/sweep_summary.py` plot): every bracket, marker shape = glide plane, open marker = column-centred first-moved core, +28 core-change event as an open square, B2 χ = 0 as a diamond; Schmid, our Vitek fit and Chaussidon's fit.
- `figures/surrogate_chi_v1.png`: now the D4 model (gp_tau_d4) on the 12 D1 points; round 2 drawn as 2* (not fitted, pending).

## Session 14 — Wed 28 Oct 2026 (Block 3): label-free "moved" rule (D5), χ = 0 baseline (D6), live round 3, README v0.1

- **Plain words:**
  - "Moved" no longer depends on which triangle the DD finder names. The start state is lost either by a **glide jump** (the compact core arrives at the next equivalent site) or by an **easy → split transition** (the core becomes centred on one atom column). Re-checking every step of every run with this rule changed exactly one result, as expected: χ = +28 is 2.110–2.120 GPa, an easy → split transition.
  - The twinning-side points (−30, −29, −28, −24) and +28 are easy → split transitions. The split core is a known artifact of the Mendelev potential (Gilbert et al., PRL 111, 095502 (2013)), so these points describe the potential, not real iron.
  - χ = 0 now uses the same loading recipe as every other angle (1.315 GPa). With it, the curve through the (−110) glide jumps is smooth: a GP learns no extra noise there.
  - Live round 3 (χ = −29) landed inside its predicted band, but the band was wide (±0.09 GPa) because the +28 surprise had raised the model's noise level. It is a hit, but a weak test.
- **Decisions recorded:** D5–D8 (see "Decisions so far").

### 1. D5 in the code; tests
- `tools/moved_rule.py`: `classify(dy, dz, orientation, topcol)` → `none` / `glide_jump` (≥ one jump to a 'down' site with top column < 1.4 b) / `split` (top column ≥ 1.4 b) / `anomaly` (compact, shifted < one jump: logged, not moved); `same_state()` (label-free: same compact site, or same centre column within 0.1 Å); `dd_state()` (one DD-finder reader shared by all scripts). D1 kept as `classify_d1` for history.
- `tools/peierls_chi.py --moved dd` = D5 (new csv columns `topcol_b`, `topcol_dy_A`, `topcol_dz_A`, `event`); `--moved dd_d1` / `dd_s12` reproduce older runs. `tools/stability_test.py`: `returned` = D5 same-state check (old triangle check printed as `returned_triangle`). `scripts/al_live_record.py` and `scripts/audit_moves.py` classify every step with D5.
- **Tests** (`tests/test_session14.py`): split at 1.346 Å / 1.80 b is moved; −28's split with the other triangle label is moved too; the +28 2.160 → 2.170 relabelling (same column C, both split) is the same state, not a new event; a 2.331 Å jump to a compact 'down' site is a glide jump (also +30's 3 jumps); compact at the start is not moved; ratio models give identical τ_P with τ0 = 1.315 or 1.0 (D6). **Full suite: 21/21 pass.**

### 2. Audit of every step under D5 (`scripts/audit_moves.py`, `data/audit_moves.csv`; D1 version `data/audit_moves_d1.csv`)
- Every saved step of 17 runs re-analysed (≈ 700 DD maps, 2.6 min on 6 cores). **Gate: PASS.** The only bracket that changed is +28 (D1 2.160–2.170 → D5 2.110–2.120, easy → split). No anomalies anywhere, and no run returns to "unmoved" after its first move.

| χ | run | D5 bracket (GPa) | event | glide plane | top column at first move (b) |
|---|---|---|---|---|---|
| −30 | small / 2× | 1.160–1.170 / 1.170–1.180 | easy → split | — | 1.540 / 1.539 |
| −29 | live 3 | 1.160–1.170 | easy → split | — | 1.542 |
| −28 | live 1 | 1.160–1.170 | easy → split | — | 1.545 |
| −24 | small | 1.180–1.190 | easy → split | — | 1.552 |
| −18, −12, −6 | small | 1.210–1.220, 1.220–1.230, 1.250–1.260 | glide jump | (−110) | 1.13–1.21 |
| 0 | B2 (baseline) / Gate B' (history) | 1.310–1.320 / 1.285–1.290 | glide jump | (−110) | 1.130 / 1.160 |
| +6, +12, +18, +24 | small | 1.390–1.400, 1.490–1.500, 1.640–1.650, 1.880–1.890 | glide jump | (−110) | 1.13–1.22 |
| +28 | live 2 (ramp + continuation) | **2.110–2.120** | easy → split | — | 1.803 |
| +30 | small / 2× | 2.040–2.050 / 2.040–2.050 | glide jump | (−1 0 1) | 1.159 / 1.213 |

### 3. Live round 2 re-recorded under D5; +28 stability with the D5 check
- Stability at 2.110 GPa re-run with the D5 check (`sims/07_al_live/stab_p28_d5.sh`, crash-safe): **2/2 returned** (compact at the start site, 1.225 b; fmax 1.5–1.7e−8 eV/Å).
- Round 2: **2.110–2.120 GPa (mid 2.115), easy → split; pre-registered 2.0086 ± 0.0094 → z = +11.35, outside ±2** (unchanged prediction). Round 1 re-recorded under D5: unchanged (1.160–1.170, easy → split, z = +1.04). Previous versions: `data/al_live_results_d1.csv`, `_s12rule.csv`, `_recorderbug.csv`.

### 4. D6: χ = 0 baseline, and the Vitek refit (`scripts/fit_vitek.py`; Session 13 fit kept as `data/vitek_fit_s13.json`)
- `data/surrogate_train_v1.csv`: χ = 0 row = B2 (1.310–1.320, mid 1.315), new `event` column; old file kept as `data/surrogate_train_v1_gateBp.csv`. `TAU0 = 1.315` in `scripts/surrogate_common.py`; Schmid curves 1.315 / cos χ. `scripts/surrogate_loo.py` (Session 12) now writes only the `_gateBp` file, so it cannot overwrite the D6 set.
- **Vitek fit on the 8 (−110) glide jumps** (−18 … +24 with B2 at 0): **s0 = 1860.7 ± 18.2 MPa, a = 0.818 ± 0.032**, RMS residual 14.2 MPa (Session 13: s0 = 1862.4 ± 23.0, a = 0.829 ± 0.041, RMS 17.9). **a changed by −0.011 (−1.3 %), inside its error.** The χ = 0 residual fell from −29 to −5.5 MPa. Residuals (MPa): −18 +22, −12 +5, −6 −6, 0 −6, +6 −7, +12 −17, +18 −15, +24 +23. Ours / Chaussidon's fit: 1.01 (−30) to 1.15 (+24); +28 (split) lies 74 MPa above our (−110) fit; +30 (on (−1 0 1)) 104 MPa below it.

### 5. Live round 3 (`sims/07_al_live/run_round3.sh`, `chi_small_m29/`)
- **Model:** gp_tau_d4 (D7, not re-selected) refitted on the 13 corrected points (v1 with B2, rounds 1–2 under D5). Learned noise 0.038 GPa (the +28 point).
- **Pre-registered** (12:02:27): **χ = −29°, 1.1668 ± 0.0452 GPa, ±2 band 1.0764–1.2572** (runner-up +29 at the same spread 0.0452, then ±17). New row; the voided Session 12 row stays.
- Gate A calibration of −29 (D2): **8/8 PASS** (max σ_xz 0.067 % / 0.019° at the model checks).
- Ramp (D5, same settings as the sweep): **1.160–1.170 GPa, easy → split** (Burgers triangle one site at +60°, top column 1.542 b). Stability at 1.160: **2/2 returned** (fmax ≤ 2.0e−8). 6.5 min ramp, 7.5 min in total (estimate 10–14).
- **z = −0.04: inside ±2.** A weak test: the band is ±0.09 GPa, and the neighbours −30 and −28 have the same value.
- **Max pool spread:** 0.0452 before round 3 → **0.0428 after (−5 %)**, maximum now at +29. Under the same model through all rounds: 0.0216 (no live points) → 0.0047 (after round 1; noise at the floor) → 0.0452 (after round 2; noise 0.038) → 0.0428 (after round 3).

### 6. Leave-one-out on the corrected data (information only; `scripts/surrogate_loo_d4.py s14`, `data/surrogate_loo_d4_s14*`)

| Set | Model | RMSE (GPa) | ±1 / ±2 | learned noise (GPa) |
|---|---|---|---|---|
| all 14 | gp_tau_d4 | 0.0718 | 0.71 / 0.79 | 0.0360 |
| all 14 | gp_ratio_d4 | 0.0726 | 0.64 / 0.79 | 0.0325 |
| all 14 | gp_ratio (fixed) | 0.0913 | 0.71 / 0.71 | 0.0029 |
| (−110) only, 8 | gp_tau_d4 | 0.0344 | 0.62 / 0.88 | 0.0029 (floor) |
| (−110) only, 8 | gp_ratio_d4 | 0.0305 | 0.25 / 0.88 | 0.0029 (floor) |
| (−110) only, 8 | gp_ratio (fixed) | 0.0265 | 0.50 / 0.88 | 0.0029 |

- **Reading:** the all-points error is dominated by +24 (+0.133), +28 (−0.100) and +30 (+0.198 GPa): an easy → split value between two glide jumps. Elsewhere |err| ≤ 0.024 GPa. On the (−110) glide jumps alone the learned noise sits at the 0.0029 floor: **the "0.016 GPa structure" of Session 13 was mostly the χ = 0 protocol kink.** A smooth GP cannot represent two kinds of event; a model that knows the event type is a v0.2 item.

### 7. Figures and README
- `figures/peierls_vs_chi.png`: Schmid 1.315 / cos χ; markers: filled = glide jump ((−110) circle, (−1 0 1) triangle), open diamond = easy → split; Gate B' as a grey history cross; live rounds labelled AL 1–3; our Vitek fit and Chaussidon's.
- `figures/surrogate_chi_v1.png`: gp_tau_d4 (D7) on the 14 points; the three pre-registered predictions with their ±2 bars; the voided row not drawn.
- `README_draft.md` (private, git-ignored, not published): rewritten as v0.1 (question, method, validation chain, verification gates, results, the honest AL finding, limitations, how it was built). Noise numbers updated to the corrected data (0.036 GPa on all points, 0.0029 on glide jumps; ±2 coverage 0.79 / 0.88) rather than the Session 13 values in the prompt (0.016 GPa, 0.67).

---

## Things I caught (running list)
- 27 Oct: the reference a₀ = 2.8553 Å and E_coh = −4.1224 eV/atom match our own Session 1 output digit for digit. *Handled:* they were our own 20 Oct output. The references are now the paper's 2.855 Å and −4.122 eV/atom (Ansh).
- 27 Oct: `Fe_mm.eam.fs` header doesn't say which Mendelev 2003 potential it is; "potential 2" isn't confirmed by the file. *Handled (Session 4):* it is numerically identical to NIST's "Potential #2" file, version 1.
- 27 Oct: NIST also lists a 2009 revision of potential 2 (version 3, `Fe_2.eam.fs`, mass 55.85, a₀ header 2.855312). Its tables differ from ours (up to ~5 % relative in r·φ(r) and ~3 % in ρ(r), inside the bonding range r < 5.3 Å). We use version 1 (identical to the LAMMPS-distributed `Fe_mm.eam.fs`). *Handled (Ansh):* v3 not used; v1 is the published 2003 version (it reproduces the paper's elastic constants).
- 27 Oct: `log.lammps` is git-ignored, so the Session 1 log isn't in git. New checks use named logs. *Handled.*
- 27 Oct: two notes-file names were in use (`PROJECT_A_NOTES.md` vs `NOTES.md`). Renamed to `NOTES.md`. *Handled.*
- 27 Oct: the new `CLAUDE.md` names Mendelev 2003 as *the* project potential. Earlier decisions said it was only the installation check, with Proville 2012 planned. "Decisions so far" now follows the new `CLAUDE.md`. *Handled:* Ansh confirmed, see "Decisions so far".
- 27 Oct (S4): Ansh's spec put the line along z, but a single dislocation in a cell periodic along the glide direction needs a b/2 box tilt along the line. LAMMPS can only tilt along x, so the frame is relabelled: x = [111] line, y = [−110] (unchanged), z = [−1−12]. Same physics. LAMMPS's "general triclinic" mode could keep Ansh's labels, but it rotates the box internally. *Handled (Ansh):* relabelling accepted; glide stress = σ_xy.
- 27 Oct (S4): "3 repeat lengths along z" was read as 3 × a√3 (the [111] repeat of the cubic-cell lattice = 6b), which gives the expected ~28k atoms. 3 × b would give 13,986 atoms. *Handled (Ansh):* 6b accepted.
- 27 Oct (S4): **the periodic-glide (PAD) cell carries a built-in non-glide shear** of γ_xz ≈ ∓0.0097 in the two half-crystals, which is σ_xz ≈ ±0.69 GPa by a rough estimate. It cancels at the core by symmetry, but bcc screw glide is known to be sensitive to non-glide stresses, and the Peierls-stress reference is 1.21 GPa (Chaussidon et al. 2006; corrected 28 Oct from ~1.25). It scales as 1/Lz, so a longer cell along the glide direction reduces it. *Handled (Ansh):* PAD cell kept, subject to a χ = 0 Peierls-stress convergence test against a cell twice as long along glide (accept if < 3 % difference). The test itself is still to do.
- 27 Oct (S4): the "expected −4.1224 eV/atom" far-field energy in the prompt is our own bulk value; the paper's value is −4.122. Both are compared. *Handled.*
- 27 Oct (S4): inserting at the hard-core triangle doesn't give a hard core. It relaxes into the neighbouring easy core, so the easy core was identified by running both triangles. *Handled.*
- 27 Oct (S4): the core type was first checked only by our own symmetry test, not against the literature. *Handled (Ansh):* Gilbert et al., PRL 111, 095502 (2013) report a non-degenerate core for the Mendelev potential, which matches ours (read by the planning chat (Claude in claude.ai)).
- 27 Oct (S5): "every step converged" can't hold for steps above the Peierls stress, because in a cell periodic along glide the dislocation never stops. Claude applied it to the steps below the Peierls stress (all 29 converged); the 2 runaway steps were halted by design. *Flagged for Ansh.*
- 27 Oct (S5): the core tracker shows small 0.07–0.10 Å steps at some stresses, probably from atoms crossing the 0.01 eV threshold. They are far below one jump (2.331 Å), so they don't affect the Peierls bracket. *Noted.*
- 27 Oct (S6): **the screw-cell builder needed > 10 GB for the 2× cell** (whole cubic grid in memory), which is the likely cause of the WSL crashes at ~20:35/20:43. *Handled:* it now builds plane by plane (35 MB, byte-identical small cell).
- 27 Oct (S6): `nproc` reports 1 once `OMP_NUM_THREADS=1` is set, so the first launcher version started `mpirun -np 0`. *Handled:* the launcher uses `nproc --all`.
- 27 Oct (S6): `mpirun -np 8` on this 8-core laptop was 6× slower than `-np 7` (23.5 s vs 3.8 s for the same relaxation). *Handled:* NP = cores − 1.
- 27 Oct (S6): the 2× cell's built-in γ is 58 % (not ~50 %) of the small cell's. *Handled:* this is a feature of the fitting band. The unrelaxed isotropic field gives the same 0.56 ratio, and b/(2Lz) halves exactly.
- 27 Oct (S6): in the 2× cell at 1.280 GPa the core moved +2.93 Å and stopped in a new stable state (tracked height +0.45 Å). The 1.300 GPa runaway was started from the 1.250 GPa state, not from that new state. *Handled (S7):* Ansh defined the Peierls stress as the first stress that moves the core one jump or more out of its starting valley, and the DD maps show what the new state is (core centred on one column, see Session 7).
- 27 Oct (S7): **Claude misread the +0.84 eV pair-energy rise at 1.280 GPa as a sign of a different core.** It is the normal elastic rise per 10 MPa step (+0.82 to +0.84 eV every step). Claude compared against the step's start, not against the normal step-to-step trend. *Handled (Ansh's correction, checked against the CSV).*
- 27 Oct (S7): **the triangle that holds the Burgers vector does not show the core's shape.** At 1.280 GPa it sits at dy = +2.02 Å, and Claude first told Ansh (in chat only) that the core had moved one plane up. The arrow sums per column show the core is centred on column C at dy = +0.67 Å, on the glide path. *Handled:* `tools/ddmap_data.py` now prints and marks the column with the largest arrow sum.
- 27 Oct (S7): the core tracker only counts atoms within 4 Å of the original glide plane, so it under-reads any move off that plane (it reported +0.45 Å for a core centred 0.67 Å up). *Noted:* the plots now show the height so such moves are visible; for χ ≠ 0, where moves off the plane are expected, the tracker method should be revisited (Ansh).
- 27 Oct (S8): **LAMMPS `units lattice` with `orient x 1 1 1 … z -1 -1 2` gives a z spacing of ⅔ of the true repeat** (4.663 Å instead of a√6 = 6.994 Å). The first tilt-test box was therefore not periodic-compatible: 2000 atoms instead of 3000, and σ_xz = 2.55 GPa at zero strain. Caught by the zero-strain stress check. *Handled:* the box is set in Å from the true repeats, with an atom-count and smallest-distance check.
- 27 Oct (S8): **an xz tilt in the [111] frame is not a pure σ_xz.** Cubic anisotropy couples it to σ_yy and σ_zz (C'_25 = −C'_35 = 31.50 GPa = 44 % of G). The test failed Ansh's < 1 % criterion, and work stopped there. *Open, for Ansh.*
- 27 Oct (S9): **the surface-force σ_xy loading is free of non-glide stress only to first order.** In a perfect slab at 1.285 GPa the interior carries σ_xz = −0.038, σ_xx = +0.033 and σ_zz = −0.011 GPa (up to 2.9 %), growing as σ². This is the same in both cells (so the cell-size test stands). *Noted for Ansh.*
- 27 Oct (S9): **option 2 with linear strains leaves 1.4–3.3 % normal stresses** (second order). One measured correction brings them to ≤ 0.16 %. *Handled on a perfect crystal; not yet tested with a dislocation.*
- 27 Oct (S9): Claude's first textbook check of the twinning script used [111], which doesn't lie in (112). *Handled:* the in-plane ⟨111⟩ is now a parameter; the check uses [11−1] and reproduces (112)[1̄1̄1].
- 27 Oct (S9): **the energy-centroid core position under stress depends on the window size by up to ~0.8 Å,** for both the old and the new tracker. The new tracker detects jumps and moves off the plane, but gives +1.17 Å (not the DD map's +0.67 Å) for the split core at 1.280 GPa. *Open:* rely on the tracker for jump detection and on the DD finder for positions? (Ansh)
- 28 Oct (S10): Gate B failed (1.250–1.260 instead of 1.280–1.290 GPa); Claude attributed it to minimiser-path dependence (7 vs 8 ranks). ~~Open.~~ **Corrected (S11): wrong diagnosis.** The cause was the y-stretch of `change_box … remap` in `in.step_chi` (see Session 11). With the fix, 7 and 8 ranks give identical brackets.
- 28 Oct (S11): **`change_box … remap` with a shrink-wrapped y boundary stretches all y coordinates by ~10⁻⁴ on every call, even with an identical box.** Caught because a force-converged state left its valley when re-minimised with no displacement. *Handled:* `tools/rebox.py`; Gate B' rerun.
- 28 Oct (S11): Claude's `pkill -STOP -f al_seeds.py` froze Claude's own shell, because the pattern matched the shell's command line. *Handled:* pause and resume by PID.
- 28 Oct (S10): Claude launched Gate B while the calibration was still running (14 ranks on 8 cores). *Handled:* killed after ~1 min, output deleted, relaunched alone.
- 28 Oct (S10): the calibration's first interpolation rule (τ², τ³ through 2 points) failed an extra check at τ = 2.4. *Handled:* 4 calibration stresses and a τ²–τ⁴ least-squares rule; all checks pass.
- 28 Oct (S10): **the first MLP ensemble was badly overconfident** (7 % of true values inside ±1 spread) and stopped active learning at 3 points. *Handled:* bootstrap + randomized priors (63 % / 88 %), and a minimum of 5 points before stopping.
- 27 Oct (S4): the background relaxation reported "failed with exit code 1" although both runs converged. The cause was a bad `tail -2` with two files at the end of Claude's shell command; LAMMPS itself exited 0. Also fixed: `tools/run_with_progress.py` now creates `logs/` from any folder. *Handled.*

- 28 Oct (S11): **the 10 MPa refinement stepped +10, +20, +30 MPa** (it added to the updated last-unmoved stress), so χ = +30 reported 2.060 above the coarse 2.050. *Handled:* fixed base in `peierls_chi.py`; ±30 rerun.
- 28 Oct (S11): **the energy tracker missed a one-site jump at χ = −30 and −24** (it read 1.98 Å for a 2.331 Å DD jump onto (0 1 −1) at 1.20 GPa), so the first sweep reported the core's second move (1.38/1.63 GPa) as the Peierls stress. Caught because the core top column was 1.52 b at the "last unmoved" state. *Handled:* `tools/dd_scan.py` re-checked every saved step of every angle (only −30/−24 affected); "moved" now comes from the DD site (`--moved dd`); −30/−24 rerun. *Ansh to confirm the criterion change.*
- 28 Oct (S12): **Claude's live-AL recorder took the coarse "moved" step instead of the fine one** (first moved row in file order), so rounds 1–2 were recorded as 1.160–1.200 / 2.110–2.150 instead of 1.160–1.170 / 2.110–2.120 GPa, and the round-2/3 picks were fitted on wrong values. Caught because the recorded bracket did not match the driver's RESULT line. *Handled:* fixed (lowest-τ moved step), re-recorded, buggy file archived; counterfactual refit shows the round-2 miss is not due to the bug; round 3 voided.
- 28 Oct (S12): **`set -e` did not stop the live-AL driver after the χ = +3 calibration FAILED**, because the exit status was hidden by a pipe into `tee | grep`. The ramp still did not run (KeyError on the missing strain-model entry), so no step ran with an uncalibrated loading. *Handled:* `set -o pipefail`.
- 28 Oct (S12): **at χ = +28 the DD-site "moved" rule fired on a core change in place** (Burgers triangle shifted 1.346 Å along y to the neighbouring triangle; split-like core on a corner column), not a glide jump. First case where the DD-site rule and the "at least one jump" definition disagree. *Open, for Ansh.*
- 28 Oct (S12): **all surrogate models are over-confident on the real data** (LOO ±2 coverage 0.55–0.82): the measured curve wiggles by ~0.02–0.03 GPa between neighbouring angles, ~10× the bracket noise. Also, χ = 0 sits ~0.025 GPa below what every model predicts from its neighbours (kink in r at χ = 0; the only point from the Gate B' protocol). *Open, for Ansh (Next steps).*
- 28 Oct (S12): Gate A calibration fails its relative σ_xz criterion at small χ = +3 (1.23 % of a 0.026 GPa target at τ = 0.5). *Open, for Ansh.*
- 28 Oct (S11): **the per-atom energy dump written right after `minimize` had garbage for a few atoms lying exactly on z = 0** (pe ≈ 0 instead of −4.12 eV), probably because they are re-wrapped and change rank in the minimiser's cleanup. The 2× tracker then locked onto them and the 2× χ = +30 run crashed (W0 = 0). The minimisation itself was fine (same total energy; a fresh `run 0` gives normal per-atom energies). *Handled:* `run 0 post no` before `write_dump` in `in.step_chi`; scan of 644 dumps: no other valid result affected (Gate B' small np8 at 1.290 GPa was a moved step, and np7 gave the same bracket).

- 28 Oct (S13): **the Peierls reference was a restatement, not the source.** We used ~1.25 GPa (Gilbert et al. 2013 restating Chaussidon et al. 2006); Chaussidon et al. give 1210 MPa at χ = 0 for this potential (planning chat). *Handled:* every mention replaced (list in Session 13 §1); ours is +6 % (Gate B') / +8.7 % (sweep protocol).
- 28 Oct (S13): recap of Session 12 slips, as notes: the live-AL recorder took the coarse "moved" step; `set -e` without `pipefail` hid a failed calibration. *Handled (S12).*
- 28 Oct (S13): **the "moved" rule vs the definition at +28.** The S12 rule (any site change) counted an in-place core change as a move; D1 (≥ one jump to a 'down' site) then fired at 2.170 GPa on a pure relabelling (atoms moved ≤ 0.013 Å), and the stability test failed for the same reason. The same split-like state on column C counts as "moved" at −30/−28/−24. *Open, for Ansh* (Session 13 §8).
- 28 Oct (S13): **a relative tolerance on a near-zero target:** Gate A's "σ_xz within 1 %" fails at χ = +3° for a 0.037° loading tilt. *Handled:* D2 (OR loading-angle ≤ 0.1°), changed after the failure and recorded as such.
- 28 Oct (S13): **the active-learning gain was mostly "measure the ends first":** a fixed ends-then-bisect rule beats the spread rule at 5–6 points; spread wins only from 8–9 points, by ~0.001–0.002 GPa. *Noted.*
- 28 Oct (S13): **+30 lies below +28** (2.045 vs a +28 event at 2.115 GPa, and no compact jump yet at +28 up to 2.17): at +30 the core escapes on (−1 0 1), whose resolved shear equals (−110)'s there, while at +28 (−1 0 1) gets less (cos 32° vs cos 28°) and the core first goes split-like in place. *Noted;* depends on the D1 question.
- 28 Oct (S13): **the χ = 0 "kink" was a loading-protocol difference:** Gate B' ran without the calibrated χ = 0 correction; with it (sweep protocol) χ = 0 gives 1.310–1.320 GPa, on the curve. *Handled (B2); baseline choice open for Ansh.*
- 28 Oct (S13): `stability_test.py` ignored `--model` at χ = 0, so a stability test of a corrected χ = 0 ramp would have run at a different loading. *Handled* before B2 (Gate B' passed no `--model`, unchanged).

- 28 Oct (S14): **D1 depended on a label.** At +28 it fired at 2.170 GPa on a triangle relabelling with ≤ 0.013 Å of atomic motion, and called the real event at 2.120 GPa a mere core change. *Handled:* D5, label-free (glide jump or easy → split); audit of every step changed only +28.
- 28 Oct (S14): **`stability_test.py` skipped the loading model at χ = 0** (would have tested a corrected χ = 0 state at uncorrected loading). *Handled (S13)* before B2.
- 28 Oct (S14): **the χ = 0 kink was a protocol mismatch**, worth ~2 % (1.2875 → 1.315 GPa). It is the difference between our two gaps to Chaussidon's 1.21 GPa (+6.4 % with the uncorrected loading, +8.7 % with the corrected one): correcting the loading made the gap larger, not smaller. With it fixed, the (−110) glide-jump curve is smooth (learned GP noise at the floor).
- 28 Oct (S14): **the AL win disappears against bisection:** on this 1D curve a fixed ends-then-bisect rule matches or beats the spread rule (S13 §7). v0.1 shows a closed pre-registered loop, not a saving.
- 28 Oct (S14): Claude's wait loop `until ! pgrep -f "scripts/audit_moves.py"` hung because pgrep matched the loop's own command line (the same self-match as the S11 `pkill -STOP`). *Handled:* killed; wait on a file or PID instead.
- 28 Oct (S14): `scripts/surrogate_loo.py` (S12) rebuilds the training file and would have overwritten the D6 set on a re-run. *Handled:* it now writes `surrogate_train_v1_gateBp.csv`.
- 28 Oct (S14): `plot_surrogate.py` would have drawn the voided round-3 prediction next to the valid one. *Handled:* voided rows filtered.

- 28 Oct (publish): **private files were already in master's history** (the potential files, including Proville's `MCM2011_eam.fs` and the NIST copies, from the first two commits; 76 `logs/` files; the session report; step logs), so untracking them alone would still have published them with the history. *Handled:* the public repository is a single-commit orphan branch (`public`, pushed as `main`) holding only the public files; `master` keeps the full history and stays local. Also before publishing: hard-coded user paths replaced by a computed project root, private prompts redacted from the public prompt log.

---

## Decisions so far
- Runs on the laptop in WSL Ubuntu, not Colab (long runs disconnect on Colab).
- ~~Mendelev 2003 used only as the installation check.~~ **Decision (Ansh, 27 Oct): Mendelev 2003, potential 2 (`potentials/Fe_mm.eam.fs`) is the project potential for v0.1.** Why: it is already validated here (lattice constant, cohesive energy, elastic constants), and it has a published Peierls-stress reference (1.21 GPa (Chaussidon, Fivel & Rodney, *Acta Mater.* 54, 3407 (2006), Mendelev potential 2, free boundaries, 1210 MPa at χ = 0, text Section 4; read by the planning chat 28 Oct; corrected from the earlier ~1.25 GPa restated by Gilbert et al. 2013); computed here in Session 5: 1.280–1.290 GPa).
- **Proville 2012 is planned as the second potential for v0.3** (compare potentials).
- Project A goal (27 Oct, `CLAUDE.md`): PyTorch surrogate model with uncertainty, trained on LAMMPS bcc Fe simulations, with active learning.
- ~~For Project A v0.1 itself, use one of the potentials from the PhD work — planned: Proville–Rodney–Marinica (2012).~~ Superseded 27 Oct (see above). Proville–Rodney–Marinica (2012) is built for screw dislocations in iron. Gordon et al. (2011) and Chamati et al. (2006) are the other two. All are public, published potentials; no CEA data or code is used.
- Proville work is **paused** (Session 2) until the lattice-constant mismatch (file gives 2.8148 Å, header says 2.855 Å) is checked against the paper.
- **Screw-cell frame (Ansh, 27 Oct): relabelling accepted.** x = [111] (line), y = [−110] (glide-plane normal), z = [−1−12] (glide direction). Glide stress = σ_xy. Physics unchanged.
- **PAD slab geometry kept (Ansh, 27 Oct)** (standard Rodney/Proville-type cell), despite its built-in non-glide shear (γ ≈ ∓0.0097, ≈ ±0.69 GPa, opposite signs in the two half-crystals). **Condition: a cell-size convergence test before the angle sweep.** Compute the χ = 0 Peierls stress in the current cell (glide length 146.87 Å) and in a cell twice as long along the glide direction (~294 Å). Accept the small cell only if the two values differ by **less than 3 %**.
  - *Hand check (Ansh):* expected γ ≈ b/(2 L_glide) = 2.473/(2 × 146.87) = 0.0084 (isotropic PAD formula; measured 0.0096–0.0097, about 15 % higher). Stress ≈ γ × (C11 − C12 + C44)/3 = 0.0097 × 71.5 GPa = 0.69 GPa. Our elastic constants give (C11 − C12 + C44)/3 = 71.44 GPa.
- **Peierls stress definition (Ansh, 27 Oct, Session 7):** the first applied stress at which the core leaves its starting valley by at least one jump (2.331 Å), regardless of what happens afterwards. Result: 146.87 Å cell 1.280–1.290 GPa, 293.75 Å cell 1.270–1.280 GPa, difference 0.8 %.
- **Cell-size test passed (Ansh, 27 Oct):** the 146.87 Å cell is accepted for the angle sweep.
- **Plots (Ansh, 27 Oct):** Peierls-ramp plots always include the core-height (y) panel.
- **χ ≠ 0 loading (Ansh, 27 Oct, Session 8), option (a):** σ_xy = τ cos χ by surface forces (as now), and σ_xz = τ sin χ by an extra xz box tilt on top of the PAD tilt b/2: Δxz = γ·Lz with γ = σ_xz/G. The actual stress must be measured, not assumed. Rejected for v0.1: (b) a rotated crystal (only special angles fit the lattice) and (c) a cylinder with fixed boundary (Vitek/Gröger, a new cell); (c) may serve later as an independent check. **Session 8 test: the tilt also creates normal stresses** (σ_yy, σ_zz ≈ ±44 % of σ_xz in a periodic block; σ_zz ≈ −64 % and σ_xx ≈ −15 % with free y faces, by calculation). **Awaiting Ansh's decision on how to handle them.**
- **Option 2 loading APPROVED for the χ sweep (Ansh, 28 Oct)**, since Gate A passed. The "provisional" label is dropped for results that run under Gate B'.
- **Peierls-stress definition (Ansh, 28 Oct, restated):** the first applied stress at which no stable state remains in the starting valley (the core leaves it by at least one jump).
- **Bracket method upgrade (Ansh, 28 Oct; planning-chat recommendation):** force-based convergence (etol 0, ftol 1e-6 eV/Å, generous max iterations), `min_modify dmax 0.01` Å, same CG minimiser. ~~Hypothesis to test: the 1.260 GPa jump in Gate B was minimiser overshoot over a small remaining barrier.~~ Withdrawn (Ansh, 28 Oct, Session 12): the cause was the `change_box … remap` y-stretch.
- **Provisional for the overnight batch (Ansh, 27 Oct, Session 10; final approval pending):** option-2 loading with the measured correction; tracker for jumps and plane changes only, DD maps for positions and shapes; Peierls definition unchanged.
- **Core tracker for χ ≠ 0 (Ansh, 27 Oct):** a window that follows the core (centred on the last known core position each step, tracking both z and y), plus the DD-map core finder at the last stable and first moving step of every angle. Not implemented yet.
- **Screw core (Ansh, 27 Oct):** Gilbert et al., *Phys. Rev. Lett.* 111, 095502 (2013) state that the Mendelev potential gives the non-degenerate core, consistent with DFT, but also predicts a metastable split core, now known to be an artifact (read by the planning chat (Claude in claude.ai)). Our relaxed core is compact and non-degenerate, so it matches.
- **Potential file version (Ansh, 27 Oct):** NIST version 3 (2009) of potential 2 is **not used**. Our `Fe_mm.eam.fs` is identical to NIST v1 and reproduces the 2003 paper's elastic constants, so it is the published 2003 version.
- **Line length 6b (3 × a√3 = 14.84 Å) accepted (Ansh, 27 Oct).** A straight line at 0 K should not depend on line length.
- **"Moved" is decided by the DD Burgers-triangle site (Ansh, 28 Oct, Session 12)** (`tools/dd_scan.py` logic, `peierls_chi.py --moved dd`), not by the energy tracker. The tracker stays only as a runaway guard (6 Å window, halt). Results computed this way are the official ones; the `*_trackermoved` runs stay archived.
- **Withdrawn (Ansh, 28 Oct, Session 12): the planning chat's "minimiser overshoot" hypothesis and Claude's "minimiser path-dependence" claim.** Both explained the Gate B failure (1.250–1.260 instead of 1.280–1.290 GPa). The real cause was the `change_box … remap` y-stretch (Session 11 §2). With it fixed, 7 and 8 ranks give the same bracket and every resting state passes the stability test.
- **Literature status of the χ curve (Ansh, 28 Oct; planning-chat literature check; magnitude now benchmarked in Session 13 §5):** the *direction* of the asymmetry (anti-twinning side harder) matches Gröger et al. 2008, Parts II–III (read by the planning chat). The *magnitude* for the Mendelev potential is **not benchmarked** against a published curve. The size of the asymmetry varies a lot between bcc metals (large in Mo, virtually none in W; planning chat), so no expected size can be borrowed from another metal.
- **D1 (Ansh, 28 Oct, Session 13): "moved" = the DD Burgers-triangle site changed by at least one jump** (shift ≥ 2.331 − 0.01 Å, to an equivalent easy site). A smaller DD shift is logged as a **core-change event** (own stress bracket) and the ramp continues. Replaces the Session 12 rule in `peierls_chi.py --moved dd` and `al_live_record.py`. (Session 13 §8: at +28 the rule fires on a labelling flip; open question for Ansh.)
- **D2 (Ansh, 28 Oct, Session 13): Gate A σ_xz passes if within 1 % of target OR the loading-angle error |Δσ_xz|/τ ≤ 0.1° (0.001745 rad).** Other criteria unchanged. **Changed after the χ = +3 failure**, because 0.0003 GPa at τ = 0.5 GPa is only a 0.034° (measured 0.037°) tilt of the loading. It is an OR, so no earlier pass can become a fail.
- **D3 (Ansh, 28 Oct, Session 13): re-run χ = 0 with the sweep protocol** (done: 1.310–1.320 GPa, Session 13 §9).
- **D4 (Ansh, 28 Oct, Session 13): surrogate GP noise learned** (WhiteKernel) within [0.0029, 0.1] GPa, for gp_tau and gp_ratio; MLP unchanged. Fixed before any new data; the model for the rest of live active learning (gp_tau_d4, chosen by LOO RMSE, Session 13 §6).
- **Literature correction (Ansh / planning chat, 28 Oct): the χ = 0 reference is 1.21 GPa** (Chaussidon, Fivel & Rodney, *Acta Mater.* 54, 3407 (2006), Mendelev potential 2, free boundaries, 1210 MPa, text Section 4), replacing ~1.25 GPa (a restatement by Gilbert et al. 2013).
- **D5 (Ansh, 28 Oct, Session 14; replaces D1): "moved" = the compact easy core at the start site is lost**, at the first step where EITHER (a) the DD site moved by ≥ one jump (2.331 − 0.01 Å) to a 'down' site whose top column is < 1.4 b (**glide jump**), OR (b) the core becomes column-centred, top column ≥ 1.4 b (**easy → split transition**). The event type is recorded per point. Reason: Ansh's definition (no stable state left in the starting valley); the split core sits on a vertex column of the starting triangle (1.35 Å from the start), not in the starting easy-core minimum. D1 was label-dependent (the +28 "jump" at 2.170 GPa was a relabelling with ≤ 0.013 Å of motion). Stability test: returned = still compact at the same site (last unmoved state), or still centred on the same column (split state). **The twinning-side points (−24, −28, −29, −30) and +28 are easy → split transitions; the split core is a known Mendelev-potential artifact (Gilbert et al., PRL 111, 095502 (2013)), so these points characterise the potential.**
- **D6 (Ansh, 28 Oct, Session 14): χ = 0 baseline = B2** (sweep protocol, corrected loading), 1.310–1.320 GPa, midpoint 1.315; TAU0 = 1.315 everywhere (Schmid 1.315 / cos χ). Gate B' (1.2875) kept as history: uncorrected loading, −1.7° effective tilt (inference, not separately tested). The 2× cell test with the corrected loading is deferred to v0.2.
- **D7 (Ansh, 28 Oct, Session 14): live AL continues with gp_tau_d4** (selected in Session 13 A5 before these results; not re-selected); only refitted on the corrected data.
- **D8 (Ansh, 28 Oct, Session 14): the asymmetry gap (a = 0.82 vs Chaussidon's 0.61) is an open question for v0.1, not chased.** Candidate causes: our cell is periodic along the glide direction, theirs has free boundaries; the built-in PAD non-glide shear (but the 2× cell changed ±30 by only 0.0 / +0.9 %); we compare fit with fit, not with their raw data (Ansh will read their Fig. 3a).
- Elastic constants will be computed by the **static 0 K method**: tiny stretches and shears of the relaxed crystal, then measure the stress change (LAMMPS "ELASTIC" recipe). Strain sizes used: e = 0.001 and 0.005 (Session 3, Ansh). Both give the same values within 0.04 GPa.

## Next steps
- **(28 Oct, S14) v0.1 status:** all gates passed; README v0.1 drafted (private). Open for v0.2: the 2× cell test with corrected loading (D6); a = 0.82 vs 0.61 (D8; Ansh reads Chaussidon Fig. 3a); a surrogate that knows the event type (glide jump vs easy → split); two inputs (χ and a second stress component), where active learning is really tested; Proville 2012 as a second potential.
- **(28 Oct, S13) For Ansh:**
  1. ~~**The D1 question (Session 13 §8):**~~ decided: D5 (Session 14). does the split-like core on column C count as "moved"? (a) yes → +28 = 2.110–2.120 GPa, twinning side unchanged, and D1's site test needs a fix so the triangle label can't decide it; (b) no → continue the −30, −28, −24 and +28 ramps to the first compact easy core at a new site.
  2. ~~**χ = 0 baseline:**~~ decided: D6 = B2 (Session 14). Gate B' 1.285–1.290 (uncorrected loading) or B2 1.310–1.320 GPa (sweep protocol, consistent with every other angle)? Affects the Schmid curves, the ratio r and the training set. If B2, should the 2× cell test be repeated with the corrected loading?
  3. ~~Live AL round 3~~: done (Session 14 §5).
  4. Why is our asymmetry stronger than Chaussidon's (a = 0.83 vs 0.61)? Candidates: PAD slab vs their cell, loading, moved criterion.
- **(28 Oct, S12) For Ansh:**
  1. **χ = +28 (live round 2):** does a core change in place (DD site shift 1.346 Å, no jump) count as "moved"? If not, continue the +28 ramp from the 2.120 GPa state until the first real jump, and use that value.
  2. **Small-χ calibration:** the Gate A σ_xz criterion is relative (1 % of τ sin χ); at χ = +3 it failed by 0.0003 GPa absolute. Options: keep it (then add a τ = 0.5 calibration point for small χ, or exclude |χ| < 6 from the pool), or switch to an absolute floor (e.g. 1 % of τ).
  3. **Round 3** of live AL, after 1–2 (the round-3 pick will be redone on the corrected data and pre-registered again).
  4. **χ = 0 consistency:** re-run χ = 0 with the sweep protocol (ramp from 0, 50/10 MPa) to see whether the kink at χ = 0 is real or comes from the different Gate B' protocol (~10 min).
  5. **Surrogate noise:** the models are over-confident because the curve has ~0.02–0.03 GPa structure (glide-plane changes, core changes) that 0.0029 GPa noise forbids. Options: learn the noise level, or a model that allows kinks (e.g. separate pieces per glide plane). Any change must be fixed before seeing new data.
  6. Optional: retrospective AL against a simple "ends first, then bisect" rule, to see whether the spread rule adds anything beyond picking the range ends.
- **(28 Oct, S11) Open after the χ sweep:**
  - ~~Ansh: confirm "moved" from the DD site~~: confirmed (28 Oct, S12).
  - Compare the χ curve with a published one: direction matches Gröger et al. 2008 Parts II–III; magnitude not benchmarked (S12 decision 3).
  - ~~Gate A re-run after the remap discovery~~: done (S12 §1, PASS 16/16).
  - ~~Ansh: next step for the ML track~~: done (S12 §3–5).
1. ~~Install Claude Code inside Ubuntu.~~ Done (Session 2).
2. ~~Download the chosen PhD potential.~~ Done: `MCM2011_eam.fs` from NIST (Session 2).
3. Ansh: check the lattice constant (and cohesive energy and elastic constants, if given) in Proville et al. 2012 and its supplementary material. Possibly ask the authors whether `MCM2011_eam.fs` is the file used in the paper.
4. Then decide: carry on with this file, or switch to Gordon 2011 / Chamati 2006.
5. Validate the chosen potential: lattice constant, cohesive energy, elastic constants (C11, C12, C44), static 0 K method. The method is now tested on Mendelev 2003 (Session 3, all pass); reuse `checks/01_lattice` and `checks/02_elastic` with the new potential file.
6. ~~Confirm that `Fe_mm.eam.fs` is "potential 2".~~ Done (Session 4: identical to NIST v1). Ansh: optionally confirm C11, C12, C44 in the paper.
7. ~~Dislocation work: not started.~~ Session 4: one relaxed screw dislocation (easy core, compact, non-degenerate) in `sims/01_screw_cell/relaxed_down.data`. No stress applied yet.
8. ~~Ansh: axis relabelling; non-glide shear of the PAD cell.~~ Decided 27 Oct (see "Decisions so far").
8a. ~~Cell-size convergence test.~~ Done (Session 6): 146.87 Å 1.280–1.290 GPa vs 293.75 Å 1.290–1.300 GPa (driver bracket); difference ≤ 1.6 %, below 3 %. With Ansh's definition (Session 7): 1.270–1.280 GPa, difference 0.8 %.
8b. ~~Ansh: confirm the small cell.~~ Accepted (Session 7). Next: Ansh defines the χ angles for the sweep. Not started.
8c. ~~Tracker decision.~~ Decided (Session 8): a window that follows the core, plus DD at the bracket steps. To implement.
8d. **Ansh: decide how to handle the normal stresses that come with the xz tilt** (Session 8) before any χ ≠ 0 run. Option 2 is tested on perfect crystals (Session 9: passes after one measured correction), not yet decided. No dislocation run or angle sweep started.
8e. Ansh: check the twinning/anti-twinning derivation (Session 9: χ = +30° (−211) anti-twinning, χ = −30° (−12−1) twinning).
8f. ~~Tracker role~~: provisional decision (Session 10): jumps and plane changes only; DD for positions. **Changed (S11): jumps now from the DD site** (the tracker missed one at χ = −30/−24); Ansh to confirm.
8i. ~~χ sweep~~: done (Session 11): 10 angles small cell + 2× at ±30; see Session 11 §5.
8g. ~~Bracket method~~: done (Session 11): force convergence + dmax 0.01; Gate B' passed after the remap fix. Baseline χ = 0: 1.285–1.290 GPa.
8h. ~~Approve option 2~~: approved (28 Oct).
9. Next physics step (Ansh to define): apply glide stress σ_xy and find the Peierls stress (reference 1.21 GPa, Chaussidon, Fivel & Rodney 2006, read by the planning chat).
