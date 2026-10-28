#!/bin/bash
# Gate B' controls (Session 11). Launch: ../../tools/launch_safe.sh gate_bprime_controls ./run_controls.sh
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
cd $R/sims/05_gate_bprime/small
for np in 7 8; do for s in 1 2; do
  python $R/tools/stability_test.py $R/sims/05_gate_bprime/ramp_small_np7/steps/coarse_1.230.data 1.23 0 74.0301 new1.230_np${np}_seed$s --seed $s --np $np --nz 21
done; done
python $R/tools/stability_test.py $R/sims/02_peierls_chi0/steps/fine_1.260.data 1.26 0 74.0301 s1.260_nojiggle_np7 --jiggle 0 --np 7 --nz 21
python $R/tools/stability_test.py $R/sims/02_peierls_chi0/steps/coarse_1.000.data 1.00 0 74.0301 s1.000_np7_seed1 --seed 1 --np 7 --nz 21
cd $R/sims/05_gate_bprime/x2
for np in 7 8; do for s in 1 2; do
  python $R/tools/stability_test.py $R/sims/05_gate_bprime/ramp_x2_np7/steps/coarse_1.220.data 1.22 0 74.0301 x2new1.220_np${np}_seed$s --seed $s --np $np --nz 42
done; done
echo CONTROLS_DONE
