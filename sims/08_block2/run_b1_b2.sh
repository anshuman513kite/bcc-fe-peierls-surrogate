#!/bin/bash
# Session 13, B1 + B2 (one LAMMPS run at a time). Same settings as the chi sweep (sims/06_chi_sweep/run_sweep.sh), rule D1.
#   B1: chi = +28, continue from the 2.120 GPa state (core-change event) upward in 10 MPa steps to the first true jump or 2.6 GPa;
#       box/slabs from the unstrained cell (--ref), moved relative to the original easy site (--dd0 0,0); 2 stability tests at
#       the last unmoved step (in this run, or the 2.120 state of the first ramp if 2.130 already jumps).
#   B2: chi = 0 with the sweep protocol (D3): ramp 0..2.4 GPa in 50 MPa, 10 MPa refinement, calibrated chi = 0 correction
#       (strain_model_al.json "0", copied from checks/07), 2 stability tests.
# Launch: ../../tools/launch_safe.sh block2_runs ./run_b1_b2.sh   (from sims/08_block2)
set -e
set -o pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
MODEL=$R/checks/08_gateA_rebox/strain_model_al.json
REF=$R/sims/01_screw_cell/relaxed_down.data
P=$R/logs/block2_runs.log
T0=$(date +%s)
say() { echo "$(date +%H:%M:%S) [elapsed $(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $P; }
lastu() { python3 -c "import csv; r=[x for x in csv.DictReader(open('$R/data/$1.csv')) if x['moved']=='no']; print('%s_%.3f %s' % (r[-1]['kind'], float(r[-1]['tau_GPa']), r[-1]['tau_GPa']) if r else 'none none')"; }
stab() {  # name chi state tau
  for s in 1 2; do
    python $R/tools/stability_test.py $3 $4 $2 74.0301 ${1}_last_seed$s --seed $s --nz 21 --model $MODEL --ref $REF | tee -a $R/data/block2_stability.txt | tee -a $P
  done
}
# ---- B1 ----
name=chi_small_p28_cont
say "B1 (1 / 2): $name, ramp 2.13..2.60 GPa in 10 MPa from the 2.120 state (progress in logs/$name.log)"
mkdir -p $R/sims/07_al_live/$name && cd $R/sims/07_al_live/$name && cp $R/sims/01_screw_cell/screw_down.data.ref.npy $R/sims/01_screw_cell/screw_down.data.core.npy .
python $R/tools/peierls_chi.py $R/sims/07_al_live/chi_small_p28/steps/fine_2.120.data 74.0301 72.2716 $name --chi 28 --model $MODEL --ref $REF \
  --dd0 0,0 --start_core 73.5431,73.99364801952619 --coarse 2.13:2.6:0.01 --fine 0.01 --nz 21 $NEW > /dev/null
grep "RESULT\|CORE-CHANGE events" $R/logs/$name.log | tee -a $P
read lt lu <<< "$(lastu $name)"
if [ "$lt" = none ]; then state=$R/sims/07_al_live/chi_small_p28/steps/fine_2.120.data; lu=2.120; else state=steps/$lt.data; fi
say "B1 stability tests at $lu GPa ($state)"
stab $name 28 $state $lu
# ---- B2 ----
name=chi_small_0
say "B2 (2 / 2): $name, chi = 0 with the sweep protocol (progress in logs/$name.log)"
mkdir -p $R/sims/08_block2/$name && cd $R/sims/08_block2/$name && cp $R/sims/01_screw_cell/screw_down.data.ref.npy $R/sims/01_screw_cell/screw_down.data.core.npy .
python $R/tools/peierls_chi.py $REF 74.0301 72.2716 $name --chi 0 --model $MODEL --coarse 0:2.4:0.05 --fine 0.01 --nz 21 $NEW > /dev/null
grep "RESULT\|CORE-CHANGE events" $R/logs/$name.log | tee -a $P
read lt lu <<< "$(lastu $name)"
say "B2 stability tests at $lu GPa"
stab $name 0 steps/$lt.data $lu
say "BLOCK2_RUNS_DONE"
