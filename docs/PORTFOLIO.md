# CV and interview notes

**Project title:** Explainable Customer Churn MLOps Platform — Individual Portfolio Project

Suggested CV bullets:

- Built a reproducible Python churn lifecycle comparing logistic regression and CatBoost, with train/validation/test isolation, calibration assessment, capacity-based ranking metrics and global/local SHAP explanations.
- Orchestrated preparation, versioned training and independent artifact-reload evaluation with Apache Airflow TaskFlow; recorded source/split/model hashes and runtime versions for traceability.
- Packaged a schema-validated FastAPI prediction service in a non-root Docker image, added batch bounds and health checks, and configured automated tests and GitHub Actions CI using independently generated synthetic data.

Only claim hosted deployment, successful GitHub CI, or Airflow cluster execution after observing those outcomes on the actual target environment. Do not claim reduced churn, revenue uplift, production ownership, or a company role. Any synthetic benchmark number must be explicitly described as synthetic.

## Suggested interview walkthrough

1. Problem and data boundaries: what churn labels mean and when features become available.
2. Preventing leakage: preprocessing fitted only on training, validation used for candidate/threshold selection.
3. Ranking and calibration: why AP, prevalence, AUC, proper scoring rules and contact capacity all matter.
4. Model interpretation: signed log-odds contributions versus causal intervention effects.
5. Operational boundary: retraining, artifact evaluation, explicit selection, serving and future promotion policy.
6. Next improvements: future-cohort evaluation, representative base rates, registries, monitoring and randomized retention experiments.
