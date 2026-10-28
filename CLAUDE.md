# Project A — rules for Claude

## What this project is
Project A: a PyTorch surrogate model with uncertainty, trained on LAMMPS simulations of bcc Fe
(Mendelev 2003 potential, `potentials/Fe_mm.eam.fs`), with active learning.
Ansh owns the physics choices; Claude writes and runs the code.

## Start of every session
- Read this file, then `NOTES.md` (results log) and `PROMPTS.md` (prompt log) to see where things stand.

## Log every prompt first
- Log every prompt Ansh gives, **verbatim**, in `PROMPTS.md`: date, time, step name, full prompt text.
- Do this **before** starting the work.

## Software environment
- LAMMPS and Python live in the conda environment `lammps`. Use it for **every** LAMMPS or Python run.
- From a non-interactive shell, either `source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps`
  or prefix commands with `conda run -n lammps` (e.g. `conda run -n lammps lmp -in in.fe_lattice`).
- Never use the system Python or install packages outside this environment without asking.
- Potential files are kept in `potentials/`. **Never modify a potential file.**

## Physics choices belong to Ansh
- Ask before choosing or changing any **interatomic potential**, **simulation setting**
  (box size, boundary conditions, temperature, thermostat, timestep, minimiser and tolerances, dislocation geometry, etc.)
  or **target property** for the ML model.
- When asking, explain the options in plain language: what each one means physically, its pros and cons,
  and what it would change in the results. A recommendation is welcome, but the decision is Ansh's.
- Purely technical choices (file layout, plotting, code structure) do not need approval.

## Verify, never assume — numbers must be real
- Every key number is printed and compared with a published value or an independent check.
- Never report a number that was not actually computed and checked.
- Every reported result must trace to a run that was really done (log or output file in this project).
- Published/reference values must be labelled as such, with their source, and kept separate from computed ones.
- If something has not been computed yet, say so — do not estimate, guess or fill in "expected" values.

## Recording results — NOTES.md
- Record every result in `NOTES.md`: what was run, the command, the numbers, the expected value with its source, and pass/fail.
- Keep the "Things I caught" section up to date.
- Add a new dated section for each work session. Use simple techno-layman language: plain explanation first,
  technical detail (commands, files) second. Follow the existing style of the file, and update
  "Design choices", "Decisions so far" and "Next steps" as needed.

## Long runs (expected > 2 minutes)
- Before starting, print an estimated run time.
- If the estimate is over 15 minutes, remind Ansh to plug in the charger and keep the laptop lid open.
- Run the job in the background and write progress to `logs/<run_name>.log`
  (steps done / total, elapsed time, estimated time left).
- Stay available while it runs so Ansh can ask about progress.

## Crash-safe launch — every LAMMPS run (rule from 27 Oct 2026)
- Launch every LAMMPS run (short or long) with `tools/launch_safe.sh <run_name> <command...>`, which:
  - detaches it fully from the terminal (`setsid nohup ... < /dev/null &`), so a closed window or a Claude Code crash cannot kill it;
  - sends all screen output to `logs/<run_name>.out` (and LAMMPS itself runs with `-screen none -log <file>`), never to the terminal;
  - caps CPU use: `OMP_NUM_THREADS=1`; prints the core count (`nproc --all`; plain `nproc` obeys OMP_NUM_THREADS and reports 1)
    and exports `NP = cores - 1`. Use `mpirun -np $NP` at most, never all cores;
  - writes the PID to `logs/<run_name>.pid`; check liveness with `ps -p $(cat logs/<run_name>.pid)`.
- Before any build or run, check its memory need is well below the free RAM (`free -g`); the laptop has 7.6 GB.

## Git
- Commit locally after each finished step.
- Never push to GitHub (or any remote) unless Ansh says so.
