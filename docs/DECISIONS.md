# Engineering decisions

| Decision | Reason and tradeoff |
|---|---|
| Logistic baseline plus CatBoost | Establishes a simple reference while supporting nonlinear interactions and missing categorical data without a large search. |
| Train/validation/test separation | Keeps preprocessing fitting, model selection and final reporting distinct. Random splitting cannot validate temporal drift. |
| AP plus prevalence, AUC and calibration | AP describes ranking at varying recall and depends on prevalence; AUC and proper scoring rules supply complementary evidence. |
| F1 threshold plus capacity lift | Demonstrates classification while making contact-volume costs visible. No profit-optimal policy is claimed. |
| Run-scoped files and path-only XCom | Practical for a local Celery/shared-volume cluster; multi-host deployment would require common object storage. |
| Explicit artifact selection | Retraining does not silently replace the API's model. Hash verification links the selected model to its evaluation. |
| Package installed in Docker/Airflow images | Avoids import-path drift. Code changes require rebuilds; source bind mounts remain a development alternative. |
| FastAPI synchronous endpoints | Keeps model calls off the async event loop via the framework's thread pool. Requests are bounded to 100 rows; high-throughput serving is not benchmarked. |
| Synthetic public demo | Makes cloning and CI reproducible without distributing third-party assessment materials or private records/models. Synthetic performance is not real-world evidence. |
| No live cloud provisioning in core scope | Keeps the complete local lifecycle reproducible. A production host, auth gateway, registry and monitoring are explicit extensions. |

## Failure behavior

Duplicate IDs, invalid target labels, infinite numeric values, changed split hashes or mismatched model/data manifests fail execution. A missing deployment model fails API startup. Invalid API columns/types fail with HTTP 422. Data generation refuses to overwrite a source file. Model writes use atomic rename; each Airflow run has its own artifact directory. Retries are safe for unchanged input, but a source refresh during a retry should be managed upstream as a new run.
