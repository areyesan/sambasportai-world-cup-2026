#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    command = [sys.executable, str(REPO_ROOT / "scripts" / script_name)]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible tournament-update workflow.")
    parser.add_argument(
        "--stage",
        choices=["all", "evaluate", "features", "train", "predict", "validate"],
        default="all",
    )
    args = parser.parse_args()

    stages = {
        "evaluate": ["evaluate_group_stage.py"],
        "features": ["build_tournament_features.py"],
        "train": ["train_tournament_update.py"],
        "predict": ["predict_round32.py"],
        "validate": ["validate_repository.py"],
        "all": [
            "evaluate_group_stage.py",
            "build_tournament_features.py",
            "train_tournament_update.py",
            "predict_round32.py",
            "validate_repository.py",
        ],
    }
    for script in stages[args.stage]:
        run(script)


if __name__ == "__main__":
    main()
