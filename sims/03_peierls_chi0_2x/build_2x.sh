#!/bin/bash
# Cell-size convergence test, part 1: build + relax + check the 2x-glide-length screw cell (easy core, "down").
# Launch from sims/03_peierls_chi0_2x/ with:  ../../tools/launch_safe.sh build_2x ./build_2x.sh
# (launch_safe.sh activates the lammps env and exports NP = cores - 1, OMP_NUM_THREADS = 1)
set -e
export SCREW_NZ=42                                   # 42 x a*sqrt6 = 293.75 A along glide (2x the 146.87 A cell)
python ../01_screw_cell/build.py down screw_down.data
python ../../tools/run_with_progress.py screw_relax_2x 10000 -- mpirun -np $NP lmp -in ../01_screw_cell/in.relax -var tag down -log log.relax_down -screen none
grep -E "^CHECK|Stopping|Iterations" log.relax_down
PYTHONPATH=../01_screw_cell python ../01_screw_cell/analyze.py down ../../figures/core_ddmap_2x.png
python -c "import numpy as np; print('Y0 = %.4f' % np.load('screw_down.data.core.npy')[0])"
echo BUILD_2X_DONE
