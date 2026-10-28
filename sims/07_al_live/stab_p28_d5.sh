#!/bin/bash
# Session 14, step 3: stability tests at the chi = +28 last unmoved state (2.110 GPa) with the D5 label-free check.
set -e
set -o pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
cd $R/sims/07_al_live/chi_small_p28
for s in 1 2; do
  python $R/tools/stability_test.py steps/fine_2.110.data 2.11 28 74.0301 chi_small_p28_d5_last_seed$s --seed $s --nz 21 \
    --model $R/checks/08_gateA_rebox/strain_model_al.json --ref $R/sims/01_screw_cell/relaxed_down.data | tee -a $R/data/al_live_stability.txt
done
echo STAB_DONE
