from __future__ import annotations

import math

import pandas as pd

from .constants import HOST_TEAMS


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_tournament_features(
    evaluation_df: pd.DataFrame,
    team_ratings: pd.DataFrame,
    host_elo_bonus: float = 45.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-controlled rolling features from prior matches only."""
    if "trained_roster_rating" not in team_ratings.columns:
        raise ValueError("team_ratings must contain trained_roster_rating")

    start_rating = dict(zip(team_ratings["team"], team_ratings["trained_roster_rating"]))
    elo = {team: float(value) for team, value in start_rating.items()}
    stats = {team: {"mp": 0, "pts": 0, "gf": 0, "ga": 0} for team in elo}

    def per_game(record: dict, key: str) -> float:
        return record[key] / record["mp"] if record["mp"] else 0.0

    feature_rows: list[dict] = []
    ordered = evaluation_df.sort_values(["date", "group"]).copy()
    for _, row in ordered.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_stats, away_stats = stats[home], stats[away]
        is_host = (not _as_bool(row["neutral"])) and home in HOST_TEAMS
        bonus = host_elo_bonus if is_host else 0.0
        diff = (elo[home] + bonus) - elo[away]

        record = row.to_dict()
        record.update(
            {
                "elo_home_pre": elo[home],
                "elo_away_pre": elo[away],
                "elo_diff_pre": diff,
                "home_form_ppg": per_game(home_stats, "pts"),
                "away_form_ppg": per_game(away_stats, "pts"),
                "form_ppg_diff": per_game(home_stats, "pts") - per_game(away_stats, "pts"),
                "home_gf_pg": per_game(home_stats, "gf"),
                "away_gf_pg": per_game(away_stats, "gf"),
                "gf_pg_diff": per_game(home_stats, "gf") - per_game(away_stats, "gf"),
                "home_ga_pg": per_game(home_stats, "ga"),
                "away_ga_pg": per_game(away_stats, "ga"),
                "ga_pg_diff": per_game(home_stats, "ga") - per_game(away_stats, "ga"),
                "home_matches_played": home_stats["mp"],
                "away_matches_played": away_stats["mp"],
                "matchday": int(home_stats["mp"] + 1),
                "host_team_indicator": int(is_host),
            }
        )
        feature_rows.append(record)

        expected_home = 1 / (1 + 10 ** (-diff / 400))
        if row["actual_home_goals"] > row["actual_away_goals"]:
            score = 1.0
        elif row["actual_home_goals"] == row["actual_away_goals"]:
            score = 0.5
        else:
            score = 0.0
        margin_multiplier = (
            math.log(abs(row["actual_home_goals"] - row["actual_away_goals"]) + 1)
            * 2.2
            / max(1.0, abs(diff) * 0.001 + 2.2)
        )
        k = 32 * max(1.0, margin_multiplier)
        delta = k * (score - expected_home)
        elo[home] += delta
        elo[away] -= delta

        home_points = 3 if score == 1 else (1 if score == 0.5 else 0)
        away_points = 3 if score == 0 else (1 if score == 0.5 else 0)
        home_stats["mp"] += 1
        home_stats["pts"] += home_points
        home_stats["gf"] += row["actual_home_goals"]
        home_stats["ga"] += row["actual_away_goals"]
        away_stats["mp"] += 1
        away_stats["pts"] += away_points
        away_stats["gf"] += row["actual_away_goals"]
        away_stats["ga"] += row["actual_home_goals"]

    training = pd.DataFrame(feature_rows)
    final_form = []
    for team, record in stats.items():
        final_form.append(
            {
                "team": team,
                "tournament_elo": elo[team],
                "matches_played": record["mp"],
                "points": record["pts"],
                "goals_for": record["gf"],
                "goals_against": record["ga"],
                "goal_difference": record["gf"] - record["ga"],
                "points_per_game": record["pts"] / record["mp"],
                "goals_for_per_game": record["gf"] / record["mp"],
                "goals_against_per_game": record["ga"] / record["mp"],
            }
        )
    return training, pd.DataFrame(final_form)
