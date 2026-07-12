# Notebooks

These notebooks provide reproducible, presentation-oriented interfaces over the repository's version-controlled datasets and scripts.

| Notebook | Purpose |
|---|---|
| `01_group_stage_error_analysis.ipynb` | Outcome, score, goals, confidence, group, and draw-error analysis for all 72 group-stage matches. |
| `02_probability_calibration.ipynb` | Reliability diagrams, ECE, Brier/log-loss analysis, and group-held-out multinomial recalibration. |
| `03_round32_scenario_explorer.ipynb` | Round-of-32 fixture inspection, scoreline exploration, upset analysis, and Monte Carlo simulations. |
| `04_player_level_roster_update.ipynb` | Validation and aggregation pipeline for an optional official final-26 player table, with a transparent team-level fallback. |

## Launch

From the repository root:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Then open the files in `notebooks/` and run all cells from top to bottom.

## Player-level input

Notebook 04 expects an optional file at:

```text
data/raw/player_level_final26.csv
```

Use `data/raw/player_level_final26_template.csv` as the schema template. The repository does not bundle proprietary player minutes, market values, or live form feeds.

## Relationship to scripts

The notebooks do not replace the production pipeline. The canonical command remains:

```bash
python scripts/run_pipeline.py --stage all
```

The notebooks are intended for analysis, diagnostics, reporting, and scenario exploration.
