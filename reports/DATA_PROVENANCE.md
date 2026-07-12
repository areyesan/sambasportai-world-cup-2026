# Data Provenance

## Frozen baseline artifacts

The v8 files in `data/baseline/` are project-generated prediction and team-rating artifacts. They preserve the information state available immediately before the tournament.

These files contain derived team-level values, not a bundled proprietary player feed.

## Group-stage results

`data/raw/world_cup_2026_group_stage_results.csv` contains all 72 completed group-stage scores. The source name and URL are included in the file.

The scores were used for:

- post-hoc evaluation of the frozen v8 forecast,
- rolling tournament Elo updates,
- rolling points and goal-form features,
- and fitting the v9 tournament-update models.

## Processed features

`data/processed/group_stage_training_features.csv` is generated sequentially. For every match, rolling form features are calculated only from earlier matches. This prevents future group-stage results from leaking into the training row.

## Round-of-32 fixtures

`data/raw/round_of_32_fixtures.csv` contains the 16 knockout fixtures scored by the model. The prediction cutoff is stored in the prediction output.

## Restricted data

The repository does not redistribute:

- licensed club-minute feeds,
- proprietary expected-goals databases,
- paid market-value datasets,
- private injury feeds,
- or authentication credentials.

Any such future source must be stored outside Git and documented in the manifest.
