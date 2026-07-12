# Data

## Layout

- `baseline/`: frozen pre-tournament v8 prediction and rating artifacts.
- `raw/`: observed tournament results and fixtures.
- `processed/`: leakage-controlled model training features.
- `manifest/`: provenance, purpose, and licensing notes.

## Important

The repository does not bundle proprietary live player-minute feeds, paid market-value data, or private scouting data. Team-level roster features are derived artifacts from the upstream v8 package.

Every new dataset should be documented in `manifest/data_sources_manifest.csv`.
