#!/bin/bash
# chi sweep (Session 11), small cell then the 2x cell at chi = +-30. Option 2 loading (approved), corrected strains
# (checks/07_chi_calibration/strain_model.json), new minimiser settings (etol 0, ftol 1e-6 2-norm, dmax 0.01, CG),
# ramp 0..2.4 GPa (50 MPa), 10 MPa refinement; stability test (2 seeds) at the last unmoved step of every angle.
# RERUN of chi = +-30 (small cell) after the refinement-loop fix (Session 11). Waits for the main sweep (PID file) first.
# Launch: ../../tools/launch_safe.sh chi_rerun30 ./run_rerun30.sh   (from sims/06_chi_sweep)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
NEW="--etol 0 --ftol 1e-6 --dmax 0.01 --maxit 200000 --maxeval 2000000"
MODEL=$R/checks/07_chi_calibration/strain_model.json
lastu() { python3 -c "import csv; r=[x for x in csv.DictReader(open('$R/data/$1.csv')) if x['moved']=='no']; print('%s_%.3f %s' % (r[-1]['kind'], float(r[-1]['tau_GPa']), r[-1]['tau_GPa']))"; }
run_angle() {   # cell chi
  cell=$1; chi=$2
  if [ $cell = small ]; then data=$R/sims/01_screw_cell/relaxed_down.data; src=$R/sims/01_screw_cell; z0=72.2716; nz=21
  else data=$R/sims/03_peierls_chi0_2x/relaxed_down.data; src=$R/sims/03_peierls_chi0_2x; z0=146.8746; nz=42; fi
  tag=$(python3 -c "print(('m' if $chi < 0 else 'p') + str(abs($chi)))")
  name=chi_${cell}_$tag
  mkdir -p $R/sims/06_chi_sweep/$name && cd $R/sims/06_chi_sweep/$name && cp $src/screw_down.data.ref.npy $src/screw_down.data.core.npy .
  python $R/tools/peierls_chi.py $data 74.0301 $z0 $name --chi $chi --model $MODEL --coarse 0:2.4:0.05 --fine 0.01 --nz $nz $NEW
  read lt lu <<< "$(lastu $name)"
  for s in 1 2; do
    python $R/tools/stability_test.py steps/$lt.data $lu $chi 74.0301 ${name}_last_seed$s --seed $s --nz $nz --model $MODEL --ref $data
  done
}
while ps -p $(cat $R/logs/chi_sweep.pid) >/dev/null 2>&1; do sleep 20; done
for t in p30 m30; do
  n=chi_small_$t
  mv $R/sims/06_chi_sweep/$n $R/sims/06_chi_sweep/${n}_v1finebug
  mv $R/logs/$n.log $R/logs/${n}_v1finebug.log; mv $R/data/$n.csv $R/data/${n}_v1finebug.csv
  for f in $R/figures/ddmap_${n}_*.png; do mv $f ${f%.png}_v1finebug.png; done
done
for chi in 30 -30; do run_angle small $chi; done
echo RERUN30_DONE
