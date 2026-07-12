#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> None:
    required = [
        ROOT / "data/baseline/v8_match_predictions.csv",
        ROOT / "data/baseline/v8_group_advancement.csv",
        ROOT / "data/baseline/v8_final26_team_ratings.csv",
        ROOT / "data/raw/world_cup_2026_group_stage_results.csv",
        ROOT / "data/processed/group_stage_training_features.csv",
        ROOT / "models/v9/v9_outcome_model.joblib",
        ROOT / "models/v9/v9_score_models.joblib",
        ROOT / "outputs/evaluation/group_stage_evaluation_metrics.json",
        ROOT / "outputs/predictions/round_of_32_predictions.csv",
        ROOT / "ui/group_stage_review_and_round32_predictions.html",
    ]
    for path in required:
        require(path)

    baseline = pd.read_csv(ROOT / "data/baseline/v8_match_predictions.csv")
    actual = pd.read_csv(ROOT / "data/raw/world_cup_2026_group_stage_results.csv")
    r32 = pd.read_csv(ROOT / "outputs/predictions/round_of_32_predictions.csv")

    if len(baseline) != 72:
        raise AssertionError(f"Expected 72 baseline matches, found {len(baseline)}")
    if len(actual) != 72:
        raise AssertionError(f"Expected 72 actual matches, found {len(actual)}")
    if len(r32) != 16:
        raise AssertionError(f"Expected 16 Round-of-32 matches, found {len(r32)}")

    baseline_sum = baseline[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    if not np.allclose(baseline_sum, 1.0, atol=1e-6):
        raise AssertionError("Baseline probabilities do not sum to one.")

    r32_sum = r32[["p_home_win_90", "p_draw_90", "p_away_win_90"]].sum(axis=1)
    if not np.allclose(r32_sum, 1.0, atol=1e-6):
        raise AssertionError("Round-of-32 probabilities do not sum to one.")

    joblib.load(ROOT / "models/v9/v9_outcome_model.joblib")
    joblib.load(ROOT / "models/v9/v9_score_models.joblib")
    metrics = json.loads((ROOT / "outputs/evaluation/group_stage_evaluation_metrics.json").read_text())
    if metrics.get("matches_evaluated") != 72:
        raise AssertionError("Evaluation metrics do not report 72 matches.")

    print("Repository validation passed.")


if __name__ == "__main__":
    main()
