.PHONY: install evaluate features train predict pipeline validate test serve-ui

install:
	python -m pip install -e ".[dev]"

evaluate:
	python scripts/evaluate_group_stage.py

features:
	python scripts/build_tournament_features.py

train:
	python scripts/train_tournament_update.py

predict:
	python scripts/predict_round32.py

pipeline:
	python scripts/run_pipeline.py --stage all

validate:
	python scripts/validate_repository.py

test:
	pytest -q

serve-ui:
	python -m http.server 8000 --directory ui
