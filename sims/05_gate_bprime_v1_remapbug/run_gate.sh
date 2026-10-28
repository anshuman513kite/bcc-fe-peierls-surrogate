#!/bin/bash
# Gate B' (Session 11): stability tests of the Session 5 / 2x states and new chi = 0 brackets, new minimiser settings
# (etol 0, ftol 1e-6 eV/A as LAMMPS 2-norm, dmax 0.01 A, CG). All jobs one at a time; 7-rank phase first, then 8-rank phase.
# Track 2 (scripts/al_seeds.py, 1 thread) is paused (SIGSTOP) during the 8-rank phase so the 8 cores are not oversubscribed.
# Launch: ../../tools/launch_safe.sh gate_bprime ./run_gate.sh   (from sims/05_gate_bprime)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
for np in 7 8; do
  if [ $np = 8 ]; then pkill -STOP -f al_seeds.py || true; echo "PHASE np=8 (Track 2 paused)"; fi
  cd $R/sims/05_gate_bprime/small
  for t in 1.260 1.270 1.280; do for s in 1 2; do
    python $R/tools/stability_test.py $R/sims/02_peierls_chi0/steps/fine_$t.data $t 0 74.0301 s${t}_np${np}_seed$s --seed $s --np $np --nz 21
  done; done
  cd $R/sims/05_gate_bprime/x2
  for t in 1.260 1.270; do for s in 1 2; do
    python $R/tools/stability_test.py $R/sims/03_peierls_chi0_2x/steps/fine_$t.data $t 0 74.0301 x2_${t}_np${np}_seed$s --seed $s --np $np --nz 42
  done; done
  cd $R/sims/05_gate_bprime/ramp_small_np$np
  python $R/tools/peierls_chi.py $R/sims/02_peierls_chi0/steps/coarse_1.200.data 74.0301 72.2716 gateBp_small_np$np --chi 0 --nocorr \
         --coarse 1.20:1.32:0.01 --fine 0.005 --np $np --nz 21 $NEW
  cd $R/sims/05_gate_bprime/ramp_x2_np$np
  python $R/tools/peierls_chi.py $R/sims/03_peierls_chi0_2x/steps/coarse_1.200.data 74.0301 146.8746 gateBp_x2_np$np --chi 0 --nocorr \
         --coarse 1.20:1.32:0.01 --fine 0.005 --np $np --nz 42 $NEW
done
pkill -CONT -f al_seeds.py || true
echo GATE_BPRIME_DONE
