# Reproducibility Guide

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Full workflow

```bash
python scripts/run_pipeline.py --stage all
```

This runs:

1. group-stage evaluation,
2. sequential feature construction,
3. tournament-update model training,
4. Round-of-32 prediction,
5. artifact validation.

## Individual commands

```bash
python scripts/evaluate_group_stage.py
python scripts/build_tournament_features.py
python scripts/train_tournament_update.py
python scripts/predict_round32.py
python scripts/validate_repository.py
pytest -q
```

## Expected row counts

- 72 pre-tournament group-stage predictions.
- 72 actual group-stage results.
- 72 leakage-controlled training rows.
- 48 final team-form rows.
- 16 Round-of-32 predictions.

## Determinism

The current models use deterministic scikit-learn solvers. Re-running the pipeline with the same package versions and input CSVs should reproduce the packaged outputs to normal floating-point tolerance.

## Windows example

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts\run_pipeline.py --stage all
```
