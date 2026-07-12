#!/usr/bin/env python3
"""
Build player-level features for v8.

Inputs:
  data/external/official_final_rosters_clean.csv
  optional licensed/public player stats table with minutes, market value, goals, assists, xG, xA, defensive actions, etc.

Because market values and live club-minute feeds often have licensing restrictions,
this script is designed to merge any locally provided licensed file rather than
shipping proprietary data in the artifact.

Expected optional columns in player_stats file:
  player, club, minutes_2025_26, market_value_eur, goals, assists, xg, xa,
  progressive_passes, tackles_interceptions, saves, position

Outputs:
  data/processed/player_level_features.csv
  data/processed/team_player_aggregates.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters", default=str(EXT / "official_final_rosters_clean.csv"))
    ap.add_argument("--player-stats", default=None, help="Optional licensed/public player statistics CSV")
    args = ap.parse_args()

    rosters = pd.read_csv(args.rosters)
    rosters["player_key"] = rosters["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

    if args.player_stats:
        stats = pd.read_csv(args.player_stats)
        stats["player_key"] = stats["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
        df = rosters.merge(stats, on="player_key", how="left", suffixes=("", "_stats"))
    else:
        df = rosters.copy()
        for col in ["minutes_2025_26", "market_value_eur", "goals", "assists", "xg", "xa",
                    "progressive_passes", "tackles_interceptions", "saves"]:
            df[col] = np.nan

    # Conservative feature defaults when optional player feed is missing.
    df["minutes_2025_26"] = pd.to_numeric(df["minutes_2025_26"], errors="coerce").fillna(0)
    df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0)
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0)
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0)
    df["xa"] = pd.to_numeric(df["xa"], errors="coerce").fillna(0)
    df["player_form_score"] = (
        0.002 * df["minutes_2025_26"]
        + 0.18 * df["goals"]
        + 0.14 * df["assists"]
        + 0.10 * df["xg"]
        + 0.10 * df["xa"]
        + 0.00000001 * df["market_value_eur"]
    )
    df.to_csv(PROC / "player_level_features.csv", index=False)

    team_col = "team" if "team" in df.columns else None
    if team_col:
        agg = df.groupby(team_col).agg(
            roster_players=("player", "count"),
            total_minutes=("minutes_2025_26", "sum"),
            total_market_value_eur=("market_value_eur", "sum"),
            mean_player_form=("player_form_score", "mean"),
            top11_player_form=("player_form_score", lambda s: s.sort_values(ascending=False).head(11).mean()),
            top26_player_form=("player_form_score", "mean"),
        ).reset_index()
        agg.to_csv(PROC / "team_player_aggregates.csv", index=False)
        print(f"Wrote {len(agg)} team aggregates.")
    print(f"Wrote {len(df)} player feature rows.")

if __name__ == "__main__":
    main()
