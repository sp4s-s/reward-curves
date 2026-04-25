# dpo-curvature

Code for the experiment described in `SPEC.md`:

> Implicit-reward curvature predicts DPO overoptimization.

## Quick start

```bash
# 1. Create env (one-time)
conda env create -f environment.yml
conda activate dpocurv

# 2. Editable install
pip install -e .

# 3. Smoke test
python -c "import dpocurv; print(dpocurv.__version__)"
pytest tests/test_smoke.py -q
```

## Layout

See `SPEC.md` §9 for the full directory map and §10 for the implementation stages.
This repository is built up stage-by-stage; each stage adds files in a self-contained way
and ends with an import-check + targeted test.

## Compute

Single 1×A100-80G, ~10 wall-clock hours total. Distributed training is out of scope.
