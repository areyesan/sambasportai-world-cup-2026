#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sambasportai_wc2026.scoring import predict_round_of_32


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Round-of-32 predictions.")
    parser.add_argument("--fixtures", type=Path, default=REPO_ROOT / "data/raw/round_of_32_fixtures.csv")
    parser.add_argument("--ratings", type=Path, default=REPO_ROOT / "data/baseline/v8_final26_team_ratings.csv")
    parser.add_argument("--form", type=Path, default=REPO_ROOT / "outputs/form/team_form_after_group_stage.csv")
    parser.add_argument("--actual-results", type=Path, default=REPO_ROOT / "data/raw/world_cup_2026_group_stage_results.csv")
    parser.add_argument("--outcome-model", type=Path, default=REPO_ROOT / "models/v9/v9_outcome_model.joblib")
    parser.add_argument("--score-model", type=Path, default=REPO_ROOT / "models/v9/v9_score_models.joblib")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/predictions/round_of_32_predictions.csv")
    args = parser.parse_args()

    fixtures = pd.read_csv(args.fixtures)
    ratings = pd.read_csv(args.ratings)
    form = pd.read_csv(args.form)
    actual = pd.read_csv(args.actual_results)
    outcome_bundle = joblib.load(args.outcome_model)
    score_bundle = joblib.load(args.score_model)

    predictions = predict_round_of_32(
        fixtures,
        ratings,
        form,
        actual,
        outcome_bundle,
        score_bundle,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(predictions[[
        "home_team", "away_team", "predicted_advancing_team",
        "p_home_advance", "p_away_advance", "expected_score"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
