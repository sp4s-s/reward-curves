#!/usr/bin/env bash
# Stage-0 verification. Run inside the activated `dpocurv` conda env.
# Usage: bash scripts/00_env_check.sh

set -euo pipefail

echo "[1/5] Python version"
python -V

echo "[2/5] Editable install"
pip install -e . --quiet

echo "[3/5] Package import"
python -c "import dpocurv, sys; print('dpocurv', dpocurv.__version__, 'on', sys.version.split()[0])"

echo "[4/5] Submodule import"
python -c "from dpocurv import types, data, models, training, eval as deval, cli; \
from dpocurv.utils import seed, logging as ulog, checkpoint, io; \
print('submodules ok')"

echo "[5/5] Smoke tests"
pytest -q tests/test_smoke.py

echo "stage-0 ok"
