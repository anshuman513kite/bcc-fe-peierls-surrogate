#!/bin/bash
# Second batch: sxy sign/size check in the slab, and option-2 strains after one measured correction (correct.py).
# Launch via: ../../tools/launch_safe.sh pure_shear2 ./run2.sh
set -e
run() {  # tag geom exx eyy ezz gyz gxz gxy sxy
  mpirun -np $NP lmp -in in.block -log log.$1 -screen none -var tag $1 -var geom $2 -var exx $3 -var eyy $4 -var ezz $5 \
         -var gyz $6 -var gxz $7 -var gxy $8 -var sxy $9
  grep -E '^INFO|^GLOBAL_GPa|^INTERIOR_GPa' log.$1
}
run slab_sxy-1    slab 0 0 0 0 0 0 -1.0
run slab_sxy0.5   slab 0 0 0 0 0 0 0.5
run slab_sxy1.285 slab 0 0 0 0 0 0 1.285
run block_pure0.357c block -0.00000969 -0.00097404 0.00096011 0.00000000 0.00581600 0.00000000 0
run block_pure0.714c block -0.00003906 -0.00192206 0.00186790 0.00000000 0.01152744 0.00000000 0
run slab_pure0.357c slab -0.00000963 0.00000000 0.00096119 0.00000000 0.00582188 0.00000000 0
run slab_pure0.714c slab -0.00003896 0.00000000 0.00187212 0.00000000 0.01155137 0.00000000 0
echo PURE2_DONE
