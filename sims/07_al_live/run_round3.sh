#!/bin/bash
# Live active learning round 3 (Session 14, Ansh's D5-D7): pick with gp_tau_d4 refitted on the corrected set (D6 chi = 0 baseline,
# rounds 1-2 recorded under D5) and pre-register; Gate A calibration of the angle (D2; skipped if already in the strain model);
# small-cell ramp as the sweep (rule D5); 2 stability tests (D5 check) at the last unmoved step; record; pool spread after.
# Launch: ../../tools/launch_safe.sh al_round3 ./run_round3.sh   (from sims/07_al_live)
set -e
set -o pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
MODEL=$R/checks/08_gateA_rebox/strain_model_al.json
REF=$R/sims/01_screw_cell/relaxed_down.data
ML="$HOME/miniforge3/bin/conda run -n ml --cwd $R/scripts python -W ignore"
P=$R/logs/al_round3.log
T0=$(date +%s)
say() { echo "$(date +%H:%M:%S) [elapsed $(( ($(date +%s) - T0) / 60 )) min] $*" | tee -a $P; }
lastu() { python3 -c "import csv; r=[x for x in csv.DictReader(open('$R/data/$1.csv')) if x['moved']=='no']; print('%s_%.3f %s' % (r[-1]['kind'], float(r[-1]['tau_GPa']), r[-1]['tau_GPa']))"; }
say "round 3: pick (gp_tau_d4)"
$ML al_live_pick.py 3 | tee -a $P
chi=$(grep "^PICK 3" $P | tail -1 | awk '{print $3}')
if python3 -c "import json,sys; sys.exit(0 if '$chi' in json.load(open('$MODEL'))['correction'] else 1)"; then
  say "chi = $chi already calibrated"
else
  say "chi = $chi: Gate A calibration (D2)"
  (cd $R/checks/08_gateA_rebox && python calibrate_chi.py $chi | tee -a $P | grep GATE_A_CHI)
fi
tag=$(python3 -c "print(('m' if $chi < 0 else 'p') + str(abs($chi)))")
name=chi_small_$tag
say "ramp $name (estimate ~8-12 min; per-step progress in logs/$name.log)"
mkdir -p $R/sims/07_al_live/$name && cd $R/sims/07_al_live/$name && cp $R/sims/01_screw_cell/screw_down.data.ref.npy $R/sims/01_screw_cell/screw_down.data.core.npy .
python $R/tools/peierls_chi.py $REF 74.0301 72.2716 $name --chi $chi --model $MODEL --coarse 0:2.4:0.05 --fine 0.01 --nz 21 $NEW > /dev/null
grep "RESULT\|ANOMALY" $R/logs/$name.log | tee -a $P
read lt lu <<< "$(lastu $name)"
say "stability tests at $lu GPa"
for s in 1 2; do
  python $R/tools/stability_test.py steps/$lt.data $lu $chi 74.0301 ${name}_last_seed$s --seed $s --nz 21 --model $MODEL --ref $REF | tee -a $R/data/al_live_stability.txt | tee -a $P
done
cd $R/sims/07_al_live
python $R/scripts/al_live_record.py 3 $chi $name $R/data/al_live_stability.txt | tee -a $P
$ML al_live_pick.py 3 --evaluate | tee -a $P
say "ROUND3_DONE"
