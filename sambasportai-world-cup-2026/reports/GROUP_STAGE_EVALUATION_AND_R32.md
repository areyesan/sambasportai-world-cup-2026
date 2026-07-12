# Group-stage evaluation and Round-of-32 update

## Evaluation cutoff
The pre-tournament v8 forecasts were evaluated against all 72 completed group-stage matches.

## Headline performance
- Outcome accuracy: **59.7%** (43/72)
- Log loss: **0.880**
- Multiclass Brier score: **0.521**
- Ranked Probability Score: **0.159**
- Exact-score accuracy: **11.1%**
- Total-goals MAE: **1.64**
- Within one goal for both teams: **30.6%**
- Group winners correct: **10/12**
- Top-two team overlap: **20/24**
- Round-of-32 qualifiers captured: **26/32**

## Tournament-update model
The update uses only group-stage information available by June 27. It combines the v8 prior probabilities and xG estimates with sequential Elo updates and rolling in-tournament form. A regularized multinomial logistic model forecasts 90-minute outcomes, while two regularized log-goal regressions forecast team goals. Hyperparameters were selected with group-held-out cross-validation.

## Cross-validation
- Outcome log loss: **0.940**
- Outcome accuracy: **47.2%**
- Per-team goal MAE: **1.02**

## Interpretation
The evaluation separates outcome success from exact-score success. Correctly identifying the result class is substantially easier than identifying one exact scoreline. The Round-of-32 forecasts therefore present both 90-minute outcome probabilities and advancement probabilities after accounting for extra time and penalties.
