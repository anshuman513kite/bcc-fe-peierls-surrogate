#!/bin/bash
R=$(cd "$(dirname "$0")/../.." && pwd)      # project root (this script is two levels down)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps
cd $R/checks/08_seam_pe
mpirun -np $NP lmp -in in.seam -screen none -log log.seam_x2 -var datain $R/sims/03_peierls_chi0_2x/relaxed_down.data -var pot $R/potentials/Fe_mm.eam.fs
grep -E "^(AS_READ|SHIFTED)" log.seam_x2
echo SEAM_DONE
