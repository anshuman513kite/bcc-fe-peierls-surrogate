#!/bin/bash
# Launch via: ../../tools/launch_safe.sh tilt_xz ./run_all.sh   (from checks/03_tilt_xz)
set -e
for g in 0.0 0.005 0.010 -0.005; do
  mpirun -np $NP lmp -in in.tilt -var gamma $g -log log.tilt_$g -screen none
  grep -E '^BOX|^STRESS_GPa' log.tilt_$g
done
echo TILT_DONE
