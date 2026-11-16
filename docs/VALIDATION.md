# Validation record

Executed in the development cloud workspace on 16 November 2026. These local checks were completed before initial publication. Check [GitHub Actions](https://github.com/yipeng-913/telecom-churn-mlops/actions) for the current remote CI status.

| Check | Observed result |
|---|---|
| Unit and API tests | 15 passed: parsing, disjoint/reproducible splits, invalid IDs/labels, scoring schema, metric sanity, device-age cleaning, generator non-overwrite/reproducibility, health/metadata, unseen categories, exact SHAP reconstruction, API validation and batch limits, missing-model startup. |
| Lint | Ruff F/E9 checks passed for source, tests and scripts. |
| Full public-demo lifecycle | 6,000 independent synthetic rows; 3,600/1,200/1,200 split; pickle saved and reloaded; model/data hashes verified; metrics and SHAP generated. See RESULTS.md. |
| Selected-artifact API | The trained demo artifact was exercised through the application with the public example request; successful response and invalid-schema rejection. |
| Package distribution | Wheel and source distribution built successfully. Installed runtime requirements passed pip check. |
| Airflow 3.4 | Public DAG parsed and all three tasks reached success with DAG.test() on a separately generated 1,200-row synthetic sample. Uses local SQLite test metadata, not the user's Celery/PostgreSQL cluster. |
| Docker build | Python 3.12 slim deployment image built successfully and installed the package/dependencies. |
| Live Docker service | Runs as user `app`; healthcheck reached healthy; GET /health and /docs, actual POST /predict, SHAP additivity and HTTP 422 for invalid schema verified over HTTP. Model mounted read-only. See api-response.synthetic.json. |
| Compose | Standalone configuration validated; Airflow override service names/environment/mount inheritance checked against the previously supplied local-cluster Compose. |
| Public-file review | Private raw data, row-level outputs, original-model artifacts and assessment documents excluded. Public charts and benchmark JSON originate only from the independent synthetic run. |

The initial ordinary Docker build encountered this cloud workspace's HTTPS proxy CA trust requirement. A temporary session-only Dockerfile mounted the existing trusted CA bundle as a BuildKit secret during pip installation; TLS verification stayed enabled and no certificate entered the repository or image layers. The committed Dockerfile is the ordinary portable build for a normal registry connection. Users behind a corporate proxy may need their own approved CA trust setup.

Airflow's SQLite test emitted an async connection-cleanup warning after all tasks and the run reached success, with process exit code zero. No success is claimed for the user's separate Windows/WSL2 Docker Desktop cluster until the local instructions are run there.

The running test container was removed after verification. There is no persistent public API hosting endpoint. The GitHub workflow runs tests, synthetic integration/API smoke and image build; its live execution status is reported in GitHub Actions.
