from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import CLASSES, EPS

OUTCOME_FEATURES = [
    "logit_home_base",
    "logit_away_base",
    "expected_goals_home",
    "expected_goals_away",
    "elo_diff_pre",
    "form_ppg_diff",
    "gf_pg_diff",
    "ga_pg_diff",
    "home_matches_played",
    "away_matches_played",
    "host_team_indicator",
]

GOAL_FEATURES = [
    "expected_goals_home",
    "expected_goals_away",
    "elo_diff_pre",
    "form_ppg_diff",
    "gf_pg_diff",
    "ga_pg_diff",
    "home_matches_played",
    "away_matches_played",
    "host_team_indicator",
]


def add_base_logits(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["logit_home_base"] = np.log(
        np.clip(frame["p_home_win"], EPS, 1) / np.clip(frame["p_draw"], EPS, 1)
    )
    frame["logit_away_base"] = np.log(
        np.clip(frame["p_away_win"], EPS, 1) / np.clip(frame["p_draw"], EPS, 1)
    )
    return frame


def train_tournament_models(
    training: pd.DataFrame,
    outcome_c: float = 0.2,
    goal_alpha: float = 3.0,
    folds: int = 6,
) -> tuple[dict, dict, pd.DataFrame, dict]:
    training = add_base_logits(training)
    X = training[OUTCOME_FEATURES].astype(float)
    y = np.array([CLASSES.index(x) for x in training["actual_outcome"]])
    groups = training["group"].to_numpy()

    outcome_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=outcome_c, max_iter=1200, solver="lbfgs")),
        ]
    )
    outcome_model.fit(X, y)

    Xg = training[GOAL_FEATURES].astype(float)
    home_goal_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=goal_alpha))])
    away_goal_model = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=goal_alpha))])
    home_goal_model.fit(Xg, np.log1p(training["actual_home_goals"]))
    away_goal_model.fit(Xg, np.log1p(training["actual_away_goals"]))

    raw_home = np.maximum(0.05, np.expm1(home_goal_model.predict(Xg)))
    raw_away = np.maximum(0.05, np.expm1(away_goal_model.predict(Xg)))
    home_scale = float(training["actual_home_goals"].mean() / raw_home.mean())
    away_scale = float(training["actual_away_goals"].mean() / raw_away.mean())

    cv = GroupKFold(n_splits=folds)
    cv_probs = np.zeros((len(training), 3))
    cv_home = np.zeros(len(training))
    cv_away = np.zeros(len(training))
    records = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        om = clone(outcome_model)
        hm = clone(home_goal_model)
        am = clone(away_goal_model)
        om.fit(X.iloc[train_idx], y[train_idx])
        hm.fit(Xg.iloc[train_idx], np.log1p(training["actual_home_goals"].iloc[train_idx]))
        am.fit(Xg.iloc[train_idx], np.log1p(training["actual_away_goals"].iloc[train_idx]))
        cv_probs[valid_idx] = om.predict_proba(X.iloc[valid_idx])

        raw_train_home = np.maximum(0.05, np.expm1(hm.predict(Xg.iloc[train_idx])))
        raw_train_away = np.maximum(0.05, np.expm1(am.predict(Xg.iloc[train_idx])))
        fold_home_scale = float(
            training["actual_home_goals"].iloc[train_idx].mean() / raw_train_home.mean()
        )
        fold_away_scale = float(
            training["actual_away_goals"].iloc[train_idx].mean() / raw_train_away.mean()
        )
        cv_home[valid_idx] = (
            np.maximum(0.05, np.expm1(hm.predict(Xg.iloc[valid_idx]))) * fold_home_scale
        )
        cv_away[valid_idx] = (
            np.maximum(0.05, np.expm1(am.predict(Xg.iloc[valid_idx]))) * fold_away_scale
        )
        fold_goal_mae = float(
            np.mean(
                np.r_[
                    np.abs(cv_home[valid_idx] - training["actual_home_goals"].iloc[valid_idx]),
                    np.abs(cv_away[valid_idx] - training["actual_away_goals"].iloc[valid_idx]),
                ]
            )
        )
        records.append(
            {
                "fold": fold,
                "held_out_groups": ",".join(sorted(set(groups[valid_idx]))),
                "log_loss": float(log_loss(y[valid_idx], cv_probs[valid_idx], labels=[0, 1, 2])),
                "accuracy": float(accuracy_score(y[valid_idx], np.argmax(cv_probs[valid_idx], axis=1))),
                "goal_mae": fold_goal_mae,
            }
        )

    cv_metrics = {
        "group_held_out_folds": folds,
        "group_held_out_log_loss": float(log_loss(y, cv_probs, labels=[0, 1, 2])),
        "group_held_out_accuracy": float(accuracy_score(y, np.argmax(cv_probs, axis=1))),
        "group_held_out_goal_mae": float(
            np.mean(
                np.r_[
                    np.abs(cv_home - training["actual_home_goals"]),
                    np.abs(cv_away - training["actual_away_goals"]),
                ]
            )
        ),
        "outcome_C": outcome_c,
        "goal_ridge_alpha": goal_alpha,
        "home_goal_scale": home_scale,
        "away_goal_scale": away_scale,
        "note": "Group-held-out validation; final models are fitted on all 72 group-stage matches.",
    }
    outcome_bundle = {
        "model": outcome_model,
        "feature_cols": OUTCOME_FEATURES,
        "classes": CLASSES,
        "C": outcome_c,
    }
    score_bundle = {
        "home_model": home_goal_model,
        "away_model": away_goal_model,
        "feature_cols": GOAL_FEATURES,
        "alpha": goal_alpha,
        "target_transform": "log1p",
        "home_goal_scale": home_scale,
        "away_goal_scale": away_scale,
    }
    return outcome_bundle, score_bundle, pd.DataFrame(records), cv_metrics


def save_model_bundles(
    outcome_bundle: dict,
    score_bundle: dict,
    model_dir: Path,
    cv_records: pd.DataFrame,
    cv_metrics: dict,
    report_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(outcome_bundle, model_dir / "v9_outcome_model.joblib")
    joblib.dump(score_bundle, model_dir / "v9_score_models.joblib")
    cv_records.to_csv(report_dir / "cross_validation_results.csv", index=False)
    (report_dir / "updated_model_cv_metrics.json").write_text(
        json.dumps(cv_metrics, indent=2), encoding="utf-8"
    )
