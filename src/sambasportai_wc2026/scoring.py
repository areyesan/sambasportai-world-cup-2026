from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .constants import CLASSES, EPS, HOST_TEAMS


def poisson_scorelines(lambda_home: float, lambda_away: float, max_goals: int = 7) -> list[dict]:
    rows = []
    for home_goals in range(max_goals + 1):
        p_home = math.exp(-lambda_home) * lambda_home**home_goals / math.factorial(home_goals)
        for away_goals in range(max_goals + 1):
            p_away = math.exp(-lambda_away) * lambda_away**away_goals / math.factorial(away_goals)
            rows.append((home_goals, away_goals, p_home * p_away))
    rows.sort(key=lambda value: value[2], reverse=True)
    return [
        {"score": f"{home}-{away}", "probability": float(probability)}
        for home, away, probability in rows[:5]
    ]


def _poisson_outcomes(lambda_home: float, lambda_away: float, max_goals: int = 9) -> np.ndarray:
    home_win = draw = away_win = 0.0
    for home_goals in range(max_goals + 1):
        p_home = math.exp(-lambda_home) * lambda_home**home_goals / math.factorial(home_goals)
        for away_goals in range(max_goals + 1):
            p_away = math.exp(-lambda_away) * lambda_away**away_goals / math.factorial(away_goals)
            probability = p_home * p_away
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
    values = np.array([home_win, draw, away_win])
    return values / values.sum()


def predict_round_of_32(
    fixtures: pd.DataFrame,
    ratings: pd.DataFrame,
    form: pd.DataFrame,
    actual_results: pd.DataFrame,
    outcome_bundle: dict,
    score_bundle: dict,
    multinomial_weight: float = 0.72,
) -> pd.DataFrame:
    rate = ratings.set_index("team")
    current_form = form.set_index("team")
    observed_total_goals = (
        actual_results["actual_home_goals"] + actual_results["actual_away_goals"]
    ).mean()

    outcome_model = outcome_bundle["model"]
    outcome_features = outcome_bundle["feature_cols"]
    home_goal_model = score_bundle["home_model"]
    away_goal_model = score_bundle["away_model"]
    goal_features = score_bundle["feature_cols"]

    rows = []
    for _, fixture in fixtures.iterrows():
        home, away = fixture["home_team"], fixture["away_team"]
        is_host = int(home in HOST_TEAMS)
        host_bonus = 45 if is_host else 0
        rating_diff = (
            current_form.loc[home, "tournament_elo"] + host_bonus
            - current_form.loc[away, "tournament_elo"]
        )
        base_draw = float(np.clip(0.25 - 0.035 * abs(rating_diff) / 400, 0.16, 0.28))
        home_non_draw = 1 / (1 + 10 ** (-rating_diff / 400))
        base_home = (1 - base_draw) * home_non_draw
        base_away = (1 - base_draw) * (1 - home_non_draw)

        raw_home_xg = math.sqrt(
            max(0.15, rate.loc[home, "base_xg_for"])
            * max(0.15, rate.loc[away, "base_xg_against"])
        )
        raw_away_xg = math.sqrt(
            max(0.15, rate.loc[away, "base_xg_for"])
            * max(0.15, rate.loc[home, "base_xg_against"])
        )
        scale = observed_total_goals / max(0.4, raw_home_xg + raw_away_xg)
        base_xg_home = raw_home_xg * scale * (1.06 if is_host else 1.0)
        base_xg_away = raw_away_xg * scale * (0.97 if is_host else 1.0)

        feature_values = {
            "logit_home_base": math.log(max(EPS, base_home) / max(EPS, base_draw)),
            "logit_away_base": math.log(max(EPS, base_away) / max(EPS, base_draw)),
            "expected_goals_home": base_xg_home,
            "expected_goals_away": base_xg_away,
            "elo_diff_pre": rating_diff,
            "form_ppg_diff": current_form.loc[home, "points_per_game"]
            - current_form.loc[away, "points_per_game"],
            "gf_pg_diff": current_form.loc[home, "goals_for_per_game"]
            - current_form.loc[away, "goals_for_per_game"],
            "ga_pg_diff": current_form.loc[home, "goals_against_per_game"]
            - current_form.loc[away, "goals_against_per_game"],
            "home_matches_played": 3,
            "away_matches_played": 3,
            "host_team_indicator": is_host,
        }
        outcome_input = pd.DataFrame([feature_values])[outcome_features]
        goal_input = pd.DataFrame([feature_values])[goal_features]

        model_probs = outcome_model.predict_proba(outcome_input)[0]
        lambda_home = max(
            0.15,
            float(np.expm1(home_goal_model.predict(goal_input)[0]))
            * score_bundle["home_goal_scale"],
        )
        lambda_away = max(
            0.15,
            float(np.expm1(away_goal_model.predict(goal_input)[0]))
            * score_bundle["away_goal_scale"],
        )
        poisson_probs = _poisson_outcomes(lambda_home, lambda_away)
        final_probs = multinomial_weight * model_probs + (1 - multinomial_weight) * poisson_probs
        final_probs = final_probs / final_probs.sum()

        draw_share_home = 1 / (1 + 10 ** (-rating_diff / 500))
        p_home_advance = float(final_probs[0] + final_probs[1] * draw_share_home)
        p_away_advance = 1 - p_home_advance
        scorelines = poisson_scorelines(lambda_home, lambda_away)

        rows.append(
            {
                "date": fixture["date"],
                "home_team": home,
                "away_team": away,
                "city": fixture["city"],
                "country": fixture["country"],
                "p_home_win_90": float(final_probs[0]),
                "p_draw_90": float(final_probs[1]),
                "p_away_win_90": float(final_probs[2]),
                "p_home_advance": p_home_advance,
                "p_away_advance": p_away_advance,
                "expected_goals_home": lambda_home,
                "expected_goals_away": lambda_away,
                "expected_score": f"{round(lambda_home)}-{round(lambda_away)}",
                "most_likely_score_90": scorelines[0]["score"],
                "predicted_outcome_90": CLASSES[int(np.argmax(final_probs))],
                "predicted_advancing_team": home if p_home_advance >= 0.5 else away,
                "home_group_points": int(current_form.loc[home, "points"]),
                "away_group_points": int(current_form.loc[away, "points"]),
                "home_group_goal_difference": int(current_form.loc[home, "goal_difference"]),
                "away_group_goal_difference": int(current_form.loc[away, "goal_difference"]),
                "home_tournament_elo": float(current_form.loc[home, "tournament_elo"]),
                "away_tournament_elo": float(current_form.loc[away, "tournament_elo"]),
                "top_scorelines_90": json.dumps(scorelines, ensure_ascii=False),
                "prediction_cutoff": "2026-06-27 23:59 ET — group-stage data only",
                "model_note": "Updated multinomial outcome model plus calibrated goal models with dynamic Elo and rolling tournament-form features.",
            }
        )
    return pd.DataFrame(rows)
