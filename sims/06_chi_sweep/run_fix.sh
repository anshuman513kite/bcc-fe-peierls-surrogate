#!/bin/bash
# chi sweep, fix batch (Session 11): "moved" from the DD Burgers-triangle site (peierls_chi.py --moved dd, default now) and
# the per-atom dump re-evaluated before writing (in.step_chi: run 0 before write_dump; fixes the 2x crash at tau = 0.05).
# Small cell chi = -30, -24 rerun (tracker-moved runs archived as *_trackermoved: they missed a one-site jump at 1.20 GPa);
# then the 2x cell at chi = +30, -30 (first attempt crashed: W0 = 0 from the dump artifact; archived as chi_x2_p30_dumpbug).
# Same settings as run_sweep.sh otherwise. Launch: ../../tools/launch_safe.sh chi_fix ./run_fix.sh  (from sims/06_chi_sweep)
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
cd $R/sims/06_chi_sweep
arch() {  # name suffix
  [ -d $1 ] && mv $1 ${1}_$2
  [ -f $R/logs/$1.log ] && mv $R/logs/$1.log $R/logs/${1}_$2.log
  [ -f $R/data/$1.csv ] && mv $R/data/$1.csv $R/data/${1}_$2.csv
  for f in $R/figures/ddmap_$1_last_unmoved.png $R/figures/ddmap_$1_first_moved.png; do [ -f $f ] && mv $f ${f%.png}_$2.png; done
  true
}
arch chi_small_m30 trackermoved; arch chi_small_m24 trackermoved; arch chi_x2_p30 dumpbug
source <(sed -n '/^NEW=/,/^}/p' run_sweep.sh)
for chi in -30 -24; do run_angle small $chi; done
for chi in 30 -30; do run_angle x2 $chi; done
echo FIX_DONE
