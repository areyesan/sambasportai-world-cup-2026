# Contributing

Thank you for contributing to the SambaSportAI World Cup forecasting project.

## Development workflow

1. Create a feature branch.
2. Install the development environment with `pip install -e ".[dev]"`.
3. Run `python scripts/validate_repository.py`.
4. Run `pytest -q`.
5. Document any new dataset in `data/manifest/data_sources_manifest.csv`.
6. Include the information cutoff for any new tournament prediction.

## Data rules

- Do not commit proprietary player feeds, paid market-value data, or credentials.
- Keep raw sources and derived artifacts clearly separated.
- Record source URLs, access dates, licensing notes, and transformation steps.
- Avoid using match outcomes that occur after the stated prediction cutoff.

## Pull requests

Describe:
- the modeling or data change,
- the evaluation protocol,
- the expected impact,
- and any known limitations.
