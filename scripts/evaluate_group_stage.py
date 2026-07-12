#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sambasportai_wc2026.evaluation import evaluate_group_stage, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-tournament group-stage predictions.")
    parser.add_argument("--baseline-predictions", type=Path, default=REPO_ROOT / "data/baseline/v8_match_predictions.csv")
    parser.add_argument("--group-predictions", type=Path, default=REPO_ROOT / "data/baseline/v8_group_advancement.csv")
    parser.add_argument("--actual-results", type=Path, default=REPO_ROOT / "data/raw/world_cup_2026_group_stage_results.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/evaluation")
    args = parser.parse_args()

    predictions = pd.read_csv(args.baseline_predictions)
    group_predictions = pd.read_csv(args.group_predictions)
    actual = pd.read_csv(args.actual_results)

    audit, metrics = evaluate_group_stage(predictions, actual, group_predictions)
    save_evaluation(audit, metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
