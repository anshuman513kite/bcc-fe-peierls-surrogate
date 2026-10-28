#!/bin/bash
cd "$(dirname "$0")"
FX=$(python3 -c "print(1.23*14.8365740126*146.8745893139*0.0062415091/756)")
for cb in 0 8 9; do mpirun -np $NP lmp -in in.readtest -var datain ../ramp_small_np7/steps/coarse_1.230.data -var cb $cb -var fx $FX -log log.readtest_$cb -screen none; grep READTEST log.readtest_$cb | tail -1; done
head -12 ../ramp_small_np7/steps/coarse_1.230.data | grep -E 'xlo|zlo|xy'
