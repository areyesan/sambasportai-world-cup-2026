# SambaSportAI FIFA World Cup 2026 Predictor — v8 Model Card

## Purpose
v8 is a day-before, roster-aware refinement of the v7 model. It is designed to update team ratings and match predictions using the final-squad layer, late availability news, and roster-derived strength deltas.

## Important honesty note
The uploaded v7 package did not contain the raw historical international-results table or a licensed player-level feed with live club minutes, market values, and form. Therefore, this v8 package includes:
- a trained roster-strength checkpoint over the available final-squad feature matrix,
- a probability/xG calibration checkpoint over the v7 prediction surface,
- reproducible scripts for ingesting full official squad tables and optional licensed/public player-stat feeds,
- all processed datasets used to produce the packaged v8 forecasts.

It does **not** claim to bundle a proprietary player-minutes or market-value feed.

## Data used in the packaged v8 run
- `outputs/v7_match_predictions.csv`
- `outputs/v7_group_advancement.csv`
- `outputs/v7_official_roster_features.csv`
- `outputs/v7_friendlies_results.csv`
- public final-squad rules/status from the 2026 FIFA World Cup squads page and Reuters news reports

## Models trained
1. `v8_roster_strength_model.joblib`
   - estimator: StandardScaler + Ridge regression
   - rows: 48 teams
   - target: roster-strength index

2. `v8_probability_calibrator.joblib`
   - estimator: StandardScaler + MultiOutput Ridge
   - rows: 72 group matches
   - target: v7 probability surface for H/D/A, used as a calibration baseline

3. `v8_score_model.joblib`
   - estimator: StandardScaler + MultiOutput Ridge
   - rows: 72 group matches
   - target: v7 xG surface, then updated with final-squad attack/defense deltas

## Main outputs
- `outputs/v8_match_predictions.csv`
- `outputs/v8_group_advancement.csv`
- `outputs/v8_final26_team_ratings.csv`
- `outputs/v8_knockout_probabilities.csv`
- `outputs/v8_friendlies_results.csv`
- `ui/world_cup_2026_prediction_center_v8.html`

## Recommended v9
For a true player-level retrain, add:
- official FIFA structured squad list,
- club minutes from 2025–26,
- market values from a licensed source,
- player form/xG/xA/defensive actions,
- historical international match outcomes from the v2 training dataset.
