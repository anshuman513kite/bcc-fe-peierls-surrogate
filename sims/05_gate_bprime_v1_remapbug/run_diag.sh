#!/bin/bash
# Gate B' diagnosis (Session 11): where does the stability test start to pass, and does it depend on the displacement size?
# Launch: ../../tools/launch_safe.sh gate_bprime_diag ./run_diag.sh
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
cd $R/sims/05_gate_bprime/small
for t in 1.200 1.210 1.220; do
  python $R/tools/stability_test.py $R/sims/05_gate_bprime/ramp_small_np7/steps/coarse_$t.data $t 0 74.0301 new${t}_np7_seed1 --seed 1 --np 7 --nz 21
done
python $R/tools/stability_test.py $R/sims/05_gate_bprime/ramp_small_np7/steps/coarse_1.230.data 1.23 0 74.0301 new1.230_j1e-4_np7_seed1 --seed 1 --jiggle 0.0001 --np 7 --nz 21
python $R/tools/stability_test.py $R/sims/05_gate_bprime/ramp_small_np7/steps/coarse_1.230.data 1.23 0 74.0301 new1.230_nojiggle_np7 --jiggle 0 --np 7 --nz 21
echo DIAG_DONE
