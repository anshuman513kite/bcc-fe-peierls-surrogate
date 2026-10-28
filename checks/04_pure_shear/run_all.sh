#!/bin/bash
# Launch via: ../../tools/launch_safe.sh pure_shear ./run_all.sh   (from checks/04_pure_shear)
set -e
run() {  # tag geom exx eyy ezz gyz gxz gxy sxy
  mpirun -np $NP lmp -in in.block -log log.$1 -screen none -var tag $1 -var geom $2 -var exx $3 -var eyy $4 -var ezz $5 \
         -var gyz $6 -var gxz $7 -var gxy $8 -var sxy $9
  grep -E '^INFO|^GLOBAL_GPa|^INTERIOR_GPa' log.$1
}
# per GPa of pure sigma_xz (elastic_calc.py): eyy = -0.0027625, ezz = +0.0027625, gxz = +0.0164349
run block_ref        block 0 0 0 0 0 0 0
run block_gxy0.005   block 0 0 0 0 0 0.005 0
run block_pure0.357  block 0 -0.00098621 0.00098621 0 0.00586727 0 0
run block_pure0.714  block 0 -0.00197243 0.00197243 0 0.01173452 0 0
run block_pure-0.357 block 0 0.00098621 -0.00098621 0 -0.00586727 0 0
run slab_ref         slab 0 0 0 0 0 0 0
run slab_sxy1        slab 0 0 0 0 0 0 1.0
run slab_pure0.357   slab 0 0 0.00098621 0 0.00586727 0 0
run slab_pure0.714   slab 0 0 0.00197243 0 0.01173452 0 0
echo PURE_DONE
