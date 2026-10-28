#!/bin/bash
# Crash-safe launcher for every LAMMPS run (CLAUDE.md rule, 27 Oct 2026).
# Usage (from the run folder):  ../../tools/launch_safe.sh <run_name> <command...>
#   - fully detached from the terminal (setsid nohup, stdin from /dev/null): a closed window or a
#     Claude Code crash cannot kill the run
#   - all screen output (stdout + stderr) goes to logs/<run_name>.out, never to the terminal
#   - OMP_NUM_THREADS=1; NP = nproc - 1 is exported, so commands use "mpirun -np $NP" at most
#   - the PID goes to logs/<run_name>.pid; check liveness with: ps -p $(cat logs/<run_name>.pid)
# The command runs inside the conda env "lammps".
set -e
name=$1; shift
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/logs"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps
CORES=$(nproc --all)                  # --all: plain nproc obeys OMP_NUM_THREADS and would report 1
export OMP_NUM_THREADS=1
export NP=$(( CORES - 1 ))
echo "cores (nproc --all) = $CORES ; MPI ranks NP = $NP ; OMP_NUM_THREADS = $OMP_NUM_THREADS"
setsid nohup "$@" > "$root/logs/$name.out" 2>&1 < /dev/null &
echo $! > "$root/logs/$name.pid"
echo "started $name, PID $(cat "$root/logs/$name.pid"); screen output -> logs/$name.out"
