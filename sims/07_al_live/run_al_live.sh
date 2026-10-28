#!/bin/bash
# Live active learning, 3 rounds (Session 12, Part B step 5). Each round:
#   1. pick (ml env): best LOO model picks the chi with the largest spread; prediction + band pre-registered in data/al_live_predictions.csv
#   2. Gate A calibration of that chi (checks/08_gateA_rebox/calibrate_chi.py, rebox path); stop if it fails
#   3. small-cell ramp, same settings as the chi sweep (sims/06_chi_sweep/run_sweep.sh): option 2, etol 0, ftol 1e-6, dmax 0.01,
#      0..2.4 GPa in 50 MPa, 10 MPa refinement, "moved" from the DD site; 2 stability tests at the last unmoved step
#   4. record (data/al_live_results.csv); the next pick refits on the enlarged set
# Then the pool spread after round 3 (data/al_live_pool_after3.csv).
# Launch: ../../tools/launch_safe.sh al_live ./run_al_live.sh   (from sims/07_al_live)
set -e
set -o pipefail     # Session 12 fix: without it a failed calibration piped into tee/grep did not stop the script
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
MODEL=$R/checks/08_gateA_rebox/strain_model_al.json
ML="$HOME/miniforge3/bin/conda run -n ml --cwd $R/scripts python"
P=$R/logs/al_live.log
T0=$(date +%s)
say() { echo "$(date +%H:%M:%S) [elapsed $(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $P; }
lastu() { python3 -c "import csv; r=[x for x in csv.DictReader(open('$R/data/$1.csv')) if x['moved']=='no']; print('%s_%.3f %s' % (r[-1]['kind'], float(r[-1]['tau_GPa']), r[-1]['tau_GPa']))"; }
for k in 1 2 3; do
  say "round $k / 3: pick"
  chi=$($ML al_live_pick.py $k | tee -a $P | awk '/^PICK/ {print $3}')
  say "round $k / 3: chi = $chi; Gate A calibration"
  (cd $R/checks/08_gateA_rebox && python calibrate_chi.py $chi | tee -a $P | grep GATE_A_CHI)
  tag=$(python3 -c "print(('m' if $chi < 0 else 'p') + str(abs($chi)))")
  name=chi_small_$tag
  say "round $k / 3: ramp $name (estimate ~10-12 min; per-step progress in logs/$name.log)"
  mkdir -p $R/sims/07_al_live/$name && cd $R/sims/07_al_live/$name && cp $R/sims/01_screw_cell/screw_down.data.ref.npy $R/sims/01_screw_cell/screw_down.data.core.npy .
  python $R/tools/peierls_chi.py $R/sims/01_screw_cell/relaxed_down.data 74.0301 72.2716 $name --chi $chi --model $MODEL --coarse 0:2.4:0.05 --fine 0.01 --nz 21 $NEW > /dev/null
  grep RESULT $R/logs/$name.log | tee -a $P
  read lt lu <<< "$(lastu $name)"
  say "round $k / 3: stability tests at $lu GPa"
  for s in 1 2; do
    python $R/tools/stability_test.py steps/$lt.data $lu $chi 74.0301 ${name}_last_seed$s --seed $s --nz 21 --model $MODEL --ref $R/sims/01_screw_cell/relaxed_down.data | tee -a $R/data/al_live_stability.txt | tee -a $P
  done
  cd $R/sims/07_al_live
  python $R/scripts/al_live_record.py $k $chi $name $R/data/al_live_stability.txt | tee -a $P
  say "round $k / 3 done"
done
$ML al_live_pick.py 3 --evaluate | tee -a $P
say "AL_LIVE_DONE"
