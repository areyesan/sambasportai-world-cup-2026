#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sambasportai_wc2026.features import build_tournament_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled tournament features.")
    parser.add_argument("--evaluation", type=Path, default=REPO_ROOT / "outputs/evaluation/group_stage_prediction_vs_actual.csv")
    parser.add_argument("--ratings", type=Path, default=REPO_ROOT / "data/baseline/v8_final26_team_ratings.csv")
    parser.add_argument("--training-output", type=Path, default=REPO_ROOT / "data/processed/group_stage_training_features.csv")
    parser.add_argument("--form-output", type=Path, default=REPO_ROOT / "outputs/form/team_form_after_group_stage.csv")
    args = parser.parse_args()

    evaluation = pd.read_csv(args.evaluation)
    ratings = pd.read_csv(args.ratings)
    training, form = build_tournament_features(evaluation, ratings)

    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    args.form_output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.training_output, index=False)
    form.to_csv(args.form_output, index=False)
    print(f"Wrote {len(training)} training rows to {args.training_output}")
    print(f"Wrote {len(form)} team-form rows to {args.form_output}")


if __name__ == "__main__":
    main()
