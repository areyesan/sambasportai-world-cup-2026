from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def test_v9_checkpoints_load():
    outcome = joblib.load(ROOT / "models/v9/v9_outcome_model.joblib")
    score = joblib.load(ROOT / "models/v9/v9_score_models.joblib")
    assert "model" in outcome
    assert "home_model" in score
    assert "away_model" in score
