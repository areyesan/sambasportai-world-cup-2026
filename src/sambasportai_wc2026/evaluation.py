from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from .constants import ACTUAL_GROUP_ORDER, CLASSES, THIRD_PLACE_QUALIFIERS


def outcome_from_score(home_goals: int | float, away_goals: int | float) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def evaluate_group_stage(
    predictions: pd.DataFrame,
    actual_results: pd.DataFrame,
    group_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate pre-tournament predictions against all completed group matches."""
    merge_cols = ["group", "home_team", "away_team"]
    df = predictions.merge(actual_results, on=merge_cols, how="left", validate="one_to_one")
    if df["actual_home_goals"].isna().any():
        missing = df.loc[df["actual_home_goals"].isna(), merge_cols]
        raise ValueError(f"Missing actual scores for:\n{missing.to_string(index=False)}")

    df["actual_outcome"] = [
        outcome_from_score(h, a)
        for h, a in zip(df["actual_home_goals"], df["actual_away_goals"])
    ]
    probs = df[["p_home_win", "p_draw", "p_away_win"]].to_numpy(float)
    y_idx = np.array([CLASSES.index(x) for x in df["actual_outcome"]])
    y_one = np.eye(3)[y_idx]
    pred_idx = np.argmax(probs, axis=1)

    df["predicted_outcome_argmax"] = np.array(CLASSES)[pred_idx]
    df["outcome_correct"] = (df["actual_outcome"] == df["predicted_outcome_argmax"]).astype(int)
    df["actual_score"] = (
        df["actual_home_goals"].astype(int).astype(str)
        + "-"
        + df["actual_away_goals"].astype(int).astype(str)
    )
    df["exact_score_correct"] = (df["predicted_exact_score"] == df["actual_score"]).astype(int)
    df["expected_score_correct"] = (df["expected_score"] == df["actual_score"]).astype(int)
    df["actual_total_goals"] = df["actual_home_goals"] + df["actual_away_goals"]
    df["predicted_total_xg"] = df["expected_goals_home"] + df["expected_goals_away"]
    df["total_goals_abs_error"] = (df["predicted_total_xg"] - df["actual_total_goals"]).abs()
    df["home_goals_abs_error"] = (df["expected_goals_home"] - df["actual_home_goals"]).abs()
    df["away_goals_abs_error"] = (df["expected_goals_away"] - df["actual_away_goals"]).abs()
    df["within_one_goal_both"] = (
        (df["home_goals_abs_error"] <= 1) & (df["away_goals_abs_error"] <= 1)
    ).astype(int)
    df["actual_over_2_5"] = (df["actual_total_goals"] > 2.5).astype(int)
    df["pred_over_2_5"] = (df["predicted_total_xg"] > 2.5).astype(int)
    df["actual_btts"] = (
        (df["actual_home_goals"] > 0) & (df["actual_away_goals"] > 0)
    ).astype(int)
    lam_h = df["expected_goals_home"].to_numpy(float)
    lam_a = df["expected_goals_away"].to_numpy(float)
    df["p_btts"] = 1 - np.exp(-lam_h) - np.exp(-lam_a) + np.exp(-(lam_h + lam_a))
    df["pred_btts"] = (df["p_btts"] >= 0.5).astype(int)

    acc = accuracy_score(y_idx, pred_idx)
    ll = log_loss(y_idx, probs, labels=[0, 1, 2])
    brier = float(np.mean(np.sum((probs - y_one) ** 2, axis=1)))
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_y = np.cumsum(y_one, axis=1)[:, :-1]
    rps = float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2))

    confidence = probs.max(axis=1)
    correct = (pred_idx == y_idx).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (confidence >= lo) & (confidence < (hi if hi < 1 else 1.000001))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_idx, pred_idx, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_idx, pred_idx, labels=[0, 1, 2])

    actual_qualifiers: set[str] = set()
    for order in ACTUAL_GROUP_ORDER.values():
        actual_qualifiers.update(order[:2])
    actual_qualifiers.update(THIRD_PLACE_QUALIFIERS)

    predicted_winners = {
        group: frame.sort_values("p_group_win", ascending=False).iloc[0]["team"]
        for group, frame in group_predictions.groupby("group")
    }
    winner_hits = sum(
        predicted_winners[group] == ACTUAL_GROUP_ORDER[group][0]
        for group in ACTUAL_GROUP_ORDER
    )
    predicted_top2 = {
        group: list(frame.sort_values("p_top2", ascending=False).head(2)["team"])
        for group, frame in group_predictions.groupby("group")
    }
    top2_hits = sum(
        len(set(predicted_top2[group]) & set(ACTUAL_GROUP_ORDER[group][:2]))
        for group in ACTUAL_GROUP_ORDER
    )
    exact_top2_groups = sum(
        set(predicted_top2[group]) == set(ACTUAL_GROUP_ORDER[group][:2])
        for group in ACTUAL_GROUP_ORDER
    )
    predicted_qualifiers = set(
        group_predictions.sort_values("p_advance", ascending=False).head(32)["team"]
    )
    qualifier_hits = len(predicted_qualifiers & actual_qualifiers)

    metrics = {
        "evaluation_cutoff": "2026-06-27 after completion of the group stage",
        "matches_evaluated": int(len(df)),
        "outcome_accuracy": float(acc),
        "log_loss": float(ll),
        "multiclass_brier_score": brier,
        "ranked_probability_score": rps,
        "expected_calibration_error_10bin": float(ece),
        "exact_score_accuracy": float(df["exact_score_correct"].mean()),
        "rounded_expected_score_accuracy": float(df["expected_score_correct"].mean()),
        "home_goals_mae": float(df["home_goals_abs_error"].mean()),
        "away_goals_mae": float(df["away_goals_abs_error"].mean()),
        "total_goals_mae": float(df["total_goals_abs_error"].mean()),
        "within_one_goal_both_teams_rate": float(df["within_one_goal_both"].mean()),
        "over_2_5_accuracy": float((df["actual_over_2_5"] == df["pred_over_2_5"]).mean()),
        "btts_accuracy": float((df["actual_btts"] == df["pred_btts"]).mean()),
        "group_winner_accuracy": float(winner_hits / 12),
        "group_winners_correct": int(winner_hits),
        "top2_team_overlap_rate": float(top2_hits / 24),
        "top2_teams_correct": int(top2_hits),
        "groups_with_exact_top2_pair": int(exact_top2_groups),
        "round32_qualifier_precision_recall": float(qualifier_hits / 32),
        "round32_qualifiers_correct": int(qualifier_hits),
        "class_metrics": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist()},
    }
    return df, metrics


def save_evaluation(df: pd.DataFrame, metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "group_stage_prediction_vs_actual.csv", index=False)
    (output_dir / "group_stage_evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
