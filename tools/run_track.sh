#!/bin/bash
# Post-hoc test of tools/track_core.py on the existing chi = 0 steps (small cell and 2x cell); no minimisation.
# Launch via (from the project root): tools/launch_safe.sh track_test tools/run_track.sh
set -e
cd "$(dirname "$0")/.."
for d in 02_peierls_chi0:01_screw_cell 03_peierls_chi0_2x:03_peierls_chi0_2x; do
  sim=${d%%:*}; cell=${d##*:}
  cd sims/$sim; mkdir -p track
  for f in ../$cell/relaxed_down.data steps/*.data; do
    out=track/$(basename $f .data).pe
    [ -s $out ] || mpirun -np $NP lmp -in ../../tools/in.pe_dump -var datain $f -var dumpout $out -screen none -log none
  done
  cd ../..
done
echo TRACK_DUMPS_DONE
