#!/bin/bash
# Create the separate conda env "ml" (Track 2; the "lammps" env is not touched). PyTorch CPU-only from the PyTorch CPU wheel index.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n ml python=3.11 numpy scipy pandas matplotlib scikit-learn pytest
conda activate ml
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import sys, numpy, scipy, pandas, matplotlib, sklearn, pytest, torch; print('VERSIONS python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__, 'matplotlib', matplotlib.__version__, 'sklearn', sklearn.__version__, 'pytest', pytest.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo ML_ENV_DONE
