# Deployment guide

## A. Standalone API — recommended portfolio demo

Requires Python 3.12; Docker is optional for training and required for container deployment.

1. Install the package with `python -m pip install -e '.[dev]'` inside a virtual environment.
2. Run `make demo` once. It generates 6,000 synthetic rows, trains, evaluates, and selects the evaluated model into `runtime/model.pkl`.
3. Run `docker compose up --build -d`.
4. Open `http://localhost:8000/docs`. Check `GET /health`, then `POST /predict` using `docs/example-request.json`.
5. Use `docker compose down` to stop this project's API. Artifacts remain in the local directories.

`/model` exposes the model's exact feature contract and artifact hash. Changing the model requires explicit selection and restarting the API. Weekly training does not silently replace the running model.

To select another evaluated run:

```bash
python scripts/select_model.py --root . --run-id <run-id>
docker compose restart api
```

The selection script checks that the evaluated artifact hash matches the model. It is a deliberate local selection tool, not a business-approval or performance gate.

## B. Existing Airflow 3.4 + Celery cluster (WSL2)

The standalone `compose.yaml` is for the prediction API. Your existing `docker-compose.yaml` manages Airflow, PostgreSQL and Redis. Keep those roles separate.

Merge this repository's `src/`, `dags/`, `pyproject.toml`, `README.md`, `Dockerfile.airflow` and `compose.airflow.override.yaml` into the existing Airflow project root. Preserve its `.env`, database volume, configuration and unrelated DAGs. If data/raw/dataset.csv already exists, do not run the synthetic generator there. Copy your intended input deliberately, or generate demo data in a separate checkout.

The extension image installs this package rather than relying on a src bind mount. All relevant Airflow services build the same package. Rebuild after Python code changes. Custom dependencies in the existing Dockerfile must be merged into Dockerfile.airflow if other DAGs require them.

In the WSL Ubuntu terminal, from that root:

```bash
mkdir -p data/processed models reports
# Ensure the existing .env AIRFLOW_UID matches `id -u` and these directories are writable.
docker compose -f docker-compose.yaml -f compose.airflow.override.yaml config --quiet
docker compose -f docker-compose.yaml -f compose.airflow.override.yaml build
docker compose -f docker-compose.yaml -f compose.airflow.override.yaml up -d
docker compose -f docker-compose.yaml -f compose.airflow.override.yaml exec airflow-worker python -m pip check
docker compose -f docker-compose.yaml -f compose.airflow.override.yaml exec airflow-dag-processor airflow dags list-import-errors
```

Find `telecom_churn_pipeline` at `http://localhost:8080`, unpause and trigger. The graph has three tasks. Weekly schedule: Monday 03:00 UTC; no historical catchup. The upstream source must be refreshed before a retraining run; no automatic ingestion from an external system is implemented.

Task outputs appear under `data/processed/<run>/`, `models/<run>/` and `reports/<run>/`. XCom contains paths, so all workers need the same mounts. PostgreSQL stores orchestration metadata, not a business feature warehouse.

## C. Private datasets

Keep private work in another checkout or a separate `--root` path. The demo schema has 12 features and is independent from the private assessment data. Private schemas can contain additional features; the pipeline infers numeric/categorical columns, but expects the six EDA variables `eqpdays`, `months`, `mou_Mean`, `change_mou`, `custcare_Mean`, `rev_Mean`, plus `Customer_ID` and binary `churn`. Additional fields become predictors except ID and target. Confirm availability at prediction time; remove unsupported or prohibited features before preparing input.

The source format is semicolon-separated, decimal-comma CSV. The API contract is generated from the selected artifact, so the supplied example request applies to the synthetic model only. CLI batch scoring does not require churn labels:

```bash
churn-pipeline score --model runtime/model.pkl --input /private/new_customers.csv --output /private/scores.csv
```

Never place private data, fitted private models, row-level predictions or assessment documents under docs/. The Git ignore rules protect normal local artifact locations; they do not replace reviewing the files staged for publication.

## D. GitHub publication

The intended repository is `yipeng-913/telecom-churn-mlops`. If it does not exist and GitHub CLI is authenticated on your own machine:

```bash
gh auth login
# Run from the completed project root; commit only reviewed public source/docs.
git init -b main
git add .
git commit -m "Initial release: reproducible churn MLOps portfolio"
gh repo create yipeng-913/telecom-churn-mlops --public --source=. --remote=origin --push
```

If you received an initialized Git checkout, skip initialization/commit if already present. If the repository already exists, inspect it before pushing; do not force-push over existing work. No GitHub tokens belong in the repository or this chat.

After the actual workflows pass, tag a release (`v1.0.0`) and add repository topics: `machine-learning`, `mlops`, `airflow`, `fastapi`, `customer-churn`, `explainable-ai`. Suggested description: “Reproducible churn ML lifecycle with Airflow, SHAP explanations, a Dockerized FastAPI service and synthetic-data CI.”

## E. Hosting beyond your laptop

Build the Docker image and deploy it to a container-capable service with the artifact mounted read-only at `/app/runtime/model.pkl`. Keep its Python/ML package versions aligned with the training environment. A public hosted service also needs managed ingress authentication/TLS, appropriate resource limits and operational monitoring. These infrastructure pieces are not provisioned in this repository. GitHub source publication and a live hosted API are separate deliverables.
