#!/bin/bash
# Re-run the 2x chi=+30 tau=0 step with the patched in.step_chi (run 0 before the pe dump); check the seam atoms' pe.
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps
cd $R/checks/08_seam_pe/step0
python $R/tools/peierls_chi.py $R/sims/03_peierls_chi0_2x/relaxed_down.data 74.0301 146.8746 seamtest_x2_p30 --chi 30 \
  --model $R/checks/07_chi_calibration/strain_model.json --coarse 0:0.1:0.05 --fine 0.01 --nz 42 --ddmaps no \
  --etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000
echo STEP0_DONE
