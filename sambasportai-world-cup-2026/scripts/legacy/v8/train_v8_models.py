#!/usr/bin/env python3
"""
Train v8 roster-aware model checkpoints.

This script trains:
  - v8_roster_strength_model.joblib
  - v8_probability_calibrator.joblib
  - v8_score_model.joblib

The packaged checkpoints are trained from the v8 processed feature matrices.
For a full historical retrain, provide the historical match-level training table
from earlier project versions and extend this script to fit on actual match outcomes.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)

def main():
    team = pd.read_csv(PROC / "v8_team_training_matrix.csv")
    match = pd.read_csv(PROC / "v8_match_feature_matrix.csv")

    team_features = [
        "base_hybrid_rating","base_xg_for","base_xg_against","availability_delta",
        "attack_delta","defense_delta","goalkeeping_delta","roster_rating_delta",
        "roster_coverage_flag","squad_size_numeric"
    ]
    rating_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.8))])
    rating_model.fit(team[team_features], team["roster_strength_index"])

    match_features = [
        "base_p_home_win","base_p_draw","base_p_away_win","base_xg_home","base_xg_away",
        "rating_diff","base_rating_diff","roster_delta_diff","availability_diff",
        "attack_matchup_home","attack_matchup_away"
    ]
    prob_model = Pipeline([("scaler", StandardScaler()), ("ridge", MultiOutputRegressor(Ridge(alpha=0.3)))])
    prob_model.fit(match[match_features], match[["base_p_home_win","base_p_draw","base_p_away_win"]])

    score_model = Pipeline([("scaler", StandardScaler()), ("ridge", MultiOutputRegressor(Ridge(alpha=0.4)))])
    score_model.fit(match[match_features], match[["base_xg_home","base_xg_away"]])

    joblib.dump(rating_model, CKPT / "v8_roster_strength_model.joblib")
    joblib.dump(prob_model, CKPT / "v8_probability_calibrator.joblib")
    joblib.dump(score_model, CKPT / "v8_score_model.joblib")
    meta = {
        "team_rows": len(team),
        "match_rows": len(match),
        "team_features": team_features,
        "match_features": match_features,
        "rating_model_train_r2": float(r2_score(team["roster_strength_index"], rating_model.predict(team[team_features]))),
        "rating_model_train_mae": float(mean_absolute_error(team["roster_strength_index"], rating_model.predict(team[team_features]))),
        "note": "Packaged training is a day-before roster calibration over the v7 model surface. Full outcome retraining requires the raw historical international-results table from earlier project versions."
    }
    (CKPT / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
