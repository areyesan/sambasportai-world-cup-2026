from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_group_stage_has_72_matches():
    data = pd.read_csv(ROOT / "data/raw/world_cup_2026_group_stage_results.csv")
    assert len(data) == 72


def test_round32_has_16_matches():
    data = pd.read_csv(ROOT / "outputs/predictions/round_of_32_predictions.csv")
    assert len(data) == 16


def test_probabilities_sum_to_one():
    baseline = pd.read_csv(ROOT / "data/baseline/v8_match_predictions.csv")
    baseline_sum = baseline[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert np.allclose(baseline_sum, 1.0, atol=1e-6)

    r32 = pd.read_csv(ROOT / "outputs/predictions/round_of_32_predictions.csv")
    r32_sum = r32[["p_home_win_90", "p_draw_90", "p_away_win_90"]].sum(axis=1)
    assert np.allclose(r32_sum, 1.0, atol=1e-6)
