#!/bin/bash
# Gate B' v2 (Session 11, after the change_box-remap fix: tools/rebox.py). New minimiser settings: etol 0, ftol 1e-6 eV/A
# (LAMMPS 2-norm), dmax 0.01 A, CG. One job at a time; 7-rank phase, then 8-rank phase (Track 2 paused by PID during it).
# Launch: ../../tools/launch_safe.sh gate_bprime ./run_gate.sh   (from sims/05_gate_bprime)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
ST="python $R/tools/stability_test.py"
lastu() { python3 -c "import csv; r=[x for x in csv.DictReader(open('$R/data/$1.csv')) if x['moved']=='no']; print('%s_%.3f %s' % (r[-1]['kind'], float(r[-1]['tau_GPa']), r[-1]['tau_GPa']))"; }
cd $R/sims/05_gate_bprime/small
$ST $R/sims/02_peierls_chi0/steps/fine_1.260.data 1.26 0 74.0301 ctrl_s1.260_nojiggle_np7 --jiggle 0 --np 7 --nz 21
$ST $R/sims/02_peierls_chi0/steps/coarse_1.000.data 1.00 0 74.0301 ctrl_s1.000_np7_seed1 --seed 1 --np 7 --nz 21
for np in 7 8; do
  if [ $np = 8 ]; then kill -STOP $(cat $R/logs/al_seeds.pid) 2>/dev/null || true; echo "PHASE np=8 (Track 2 paused)"; fi
  cd $R/sims/05_gate_bprime/small
  for t in 1.260 1.270 1.280; do for s in 1 2; do
    $ST $R/sims/02_peierls_chi0/steps/fine_$t.data $t 0 74.0301 s${t}_np${np}_seed$s --seed $s --np $np --nz 21
  done; done
  cd $R/sims/05_gate_bprime/x2
  for t in 1.260 1.270; do for s in 1 2; do
    $ST $R/sims/03_peierls_chi0_2x/steps/fine_$t.data $t 0 74.0301 x2_${t}_np${np}_seed$s --seed $s --np $np --nz 42
  done; done
  cd $R/sims/05_gate_bprime/ramp_small_np$np
  python $R/tools/peierls_chi.py $R/sims/02_peierls_chi0/steps/coarse_1.200.data 74.0301 72.2716 gateBp_small_np$np --chi 0 --nocorr \
         --coarse 1.20:1.32:0.01 --fine 0.005 --np $np --nz 21 $NEW
  read tag tau <<< "$(lastu gateBp_small_np$np)"
  cd $R/sims/05_gate_bprime/small
  for s in 1 2; do $ST $R/sims/05_gate_bprime/ramp_small_np$np/steps/$tag.data $tau 0 74.0301 newlast_small_${tag}_np${np}_seed$s --seed $s --np $np --nz 21; done
  cd $R/sims/05_gate_bprime/ramp_x2_np$np
  python $R/tools/peierls_chi.py $R/sims/03_peierls_chi0_2x/steps/coarse_1.200.data 74.0301 146.8746 gateBp_x2_np$np --chi 0 --nocorr \
         --coarse 1.20:1.32:0.01 --fine 0.005 --np $np --nz 42 $NEW
  read tag tau <<< "$(lastu gateBp_x2_np$np)"
  cd $R/sims/05_gate_bprime/x2
  for s in 1 2; do $ST $R/sims/05_gate_bprime/ramp_x2_np$np/steps/$tag.data $tau 0 74.0301 newlast_x2_${tag}_np${np}_seed$s --seed $s --np $np --nz 42; done
done
kill -CONT $(cat $R/logs/al_seeds.pid) 2>/dev/null || true
echo GATE_BPRIME_DONE
