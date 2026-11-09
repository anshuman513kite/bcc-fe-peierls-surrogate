#!/bin/bash
# Steps 5 then 6 (V02_PLAN 8.11 D-v02-6 d): step 6 runs only if step 5 exits 0 (PASS at s = 0.75, rule 4 not triggered).
# Launch from checks/12_corner_gate with tools/launch_safe.sh (7 MPI ranks, one LAMMPS run at a time).
cd "$(dirname "$0")"
python run_step5.py; rc5=$?
echo "STEP5_EXIT=$rc5"
if [ $rc5 -ne 0 ]; then exit $rc5; fi
cd ../13_size_gate && python run_step6.py; rc6=$?
echo "STEP6_EXIT=$rc6"
exit $rc6
