#!/usr/bin/env python3
"""
Collect official 2026 FIFA World Cup squad tables.

This script is intended to be run in an online environment. It uses the public
Wikipedia squad page as a machine-readable mirror of FIFA-published squad lists.
For stricter production use, replace the Wikipedia URL with FIFA's official feed
if/when the project has access to a structured endpoint.

Outputs:
  data/external/official_final_rosters_raw.csv
  data/external/official_final_rosters_clean.csv
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "external"
OUT.mkdir(parents=True, exist_ok=True)

SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def main() -> None:
    tables = pd.read_html(SQUADS_URL)
    rows = []
    current_team = None
    # Wikipedia exposes multiple wikitable tables; team inference may require manual validation.
    # We keep the table_id so the downstream validation notebook can confirm mapping.
    for table_id, tab in enumerate(tables):
        df = clean_columns(tab)
        cols = set(df.columns)
        if "player" in cols and ("pos." in cols or "pos" in cols or "position" in cols):
            df["source_table_id"] = table_id
            rows.append(df)
    if not rows:
        raise RuntimeError("No player tables found. Page structure may have changed.")
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(OUT / "official_final_rosters_raw.csv", index=False)

    # Light normalization. Team names should be reviewed from table headers if using Wikipedia;
    # FIFA structured source is preferred where available.
    clean = raw.copy()
    pos_col = next((c for c in clean.columns if c in ["pos.", "pos", "position"]), None)
    if pos_col:
        clean = clean.rename(columns={pos_col: "position"})
    if "no." in clean.columns:
        clean = clean.rename(columns={"no.": "squad_number"})
    clean.to_csv(OUT / "official_final_rosters_clean.csv", index=False)
    print(f"Wrote {len(clean)} player rows to {OUT}")

if __name__ == "__main__":
    main()
