.PHONY: install test lint demo serve image
install:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
lint:
	ruff check src tests
# Fails instead of overwriting an existing dataset; use a separate checkout/root for private work.
demo:
	churn-demo --root . --rows 6000
	churn-pipeline run --root . --run-id portfolio-demo
	python scripts/select_model.py --run-id portfolio-demo
serve:
	CHURN_MODEL_PATH=runtime/model.pkl uvicorn telecom_churn.api:app --host 127.0.0.1 --port 8000
image:
	docker compose build
