# Model Card — Tournament-Updated World Cup 2026 Forecasting Model

## Model purpose

The v9 model updates the frozen pre-tournament v8 forecasts after the completion of the 2026 World Cup group stage. It predicts:

- 90-minute home win / draw / away win probabilities,
- expected home and away goals,
- most likely exact scoreline,
- and each team's probability of advancing from a knockout match.

## Information cutoff

**2026-06-27, after completion of the group stage.**

No Round-of-32 result is used in training or prediction.

## Training data

- 72 completed group-stage matches.
- Frozen v8 pre-tournament probabilities and expected goals.
- Roster-aware starting ratings from v8.
- Sequential Elo ratings updated after each prior group-stage result.
- Rolling tournament points, goals scored, and goals conceded.

## Models

### Outcome model

`StandardScaler + multinomial LogisticRegression`

Regularization parameter: `C = 0.2`

### Goal models

Two independent pipelines:

`StandardScaler + Ridge regression on log1p(goals)`

Regularization parameter: `alpha = 3.0`

The resulting goal rates are calibrated to the observed home and away scoring rates in the group stage.

## Validation

Six-fold group-held-out validation, where each fold withholds two complete groups.

- Outcome accuracy: **47.2%**
- Outcome log loss: **0.940**
- Per-team goal MAE: **1.02**

## Pre-tournament evaluation

- Result accuracy: **59.7%**
- Log loss: **0.880**
- Exact-score accuracy: **11.1%**
- Group winners: **10/12**
- Round-of-32 qualifiers captured: **26/32**

## Known limitations

- Only 72 tournament matches are available for the update.
- Draws were under-selected by the pre-tournament argmax decision rule.
- Exact-score prediction remains substantially harder than outcome prediction.
- Player-level proprietary live-form feeds are not bundled.
- Extra-time and penalty advancement probabilities are approximated using a strength-weighted split of the 90-minute draw probability.
- Match predictions are probabilistic and should not be interpreted as certainties or financial advice.
