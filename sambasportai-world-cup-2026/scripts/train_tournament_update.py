#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sambasportai_wc2026.modeling import save_model_bundles, train_tournament_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tournament-updated outcome and goal models.")
    parser.add_argument("--training-data", type=Path, default=REPO_ROOT / "data/processed/group_stage_training_features.csv")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models/v9")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    parser.add_argument("--outcome-c", type=float, default=0.2)
    parser.add_argument("--goal-alpha", type=float, default=3.0)
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    training = pd.read_csv(args.training_data)
    outcome_bundle, score_bundle, cv_records, cv_metrics = train_tournament_models(
        training,
        outcome_c=args.outcome_c,
        goal_alpha=args.goal_alpha,
        folds=args.folds,
    )
    save_model_bundles(
        outcome_bundle,
        score_bundle,
        args.model_dir,
        cv_records,
        cv_metrics,
        args.report_dir,
    )
    print(json.dumps(cv_metrics, indent=2))


if __name__ == "__main__":
    main()
