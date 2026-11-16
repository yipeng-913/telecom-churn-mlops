# Model card

**Purpose:** a reproducible portfolio demonstration of churn ranking and ML deployment. **Public data:** independently generated synthetic telecom records, seed 2026, 6,000 rows. It is not a model approved for real customer targeting.

## Target and inputs

Binary churn is sampled from a known nonlinear synthetic risk function. The generator creates device age, tenure, usage, usage change, care calls, monthly charges/revenue, dropped calls, household subscriber count, handset price, area and dualband. Approximate missingness is injected into four fields. IDs are excluded. Public synthetic inputs do not contain ethnicity, age, income or other demographic attributes.

For a private real-world churn problem, the prediction/observation window and feature timestamps must be explicitly defined; the synthetic outcome is not evidence that real features are available before churn.

## Training and evaluation

Fixed-seed stratified 60/20/20 split; unique and disjoint customer IDs. Logistic regression fits numeric imputation, scaling and categorical one-hot encoding on training only. CatBoost handles numeric missing values and explicit missing category tokens. Negative device age is normalized to missing consistently in training and scoring.

The CatBoost candidate uses depth 6, learning rate 0.05, L2 regularization 5, at most 1,200 iterations and validation-AUC early stopping. Validation AP selects the candidate; validation F1 selects the threshold. The hold-out is reserved for reporting. The threshold is not cost-optimal. Results include AP with prevalence baseline, ROC-AUC, log loss, Brier, classification metrics and top-20% precision/capture/lift.

The fixed-model bootstrap interval uses 200 row-resampling replicates. It does not cover training, feature-selection or model-selection uncertainty. Data generation, package versions and source/split/model hashes are recorded for reproducibility.

## Interpretation

CatBoost native SHAP explains raw log odds. Global importance averages absolute contributions over up to 1,000 test rows. Individual explanations retain signed contributions and baseline; additivity is checked. Correlated predictors can share credit. Neither SHAP nor observational group differences establish causal effects or treatment response.

## Limits

Synthetic metrics do not estimate real-world performance. Random hold-out does not evaluate future cohorts. No representative production base rate, intervention data, customer economics or external validation is supplied. The project includes scheduled retraining and an explicitly selected local deployment artifact; it does not include an automatic production promotion system, model registry, drift alerts, authentication gateway or online experiment platform.

Before real use: verify data rights and feature availability, choose lawful/appropriate inputs, evaluate on mature future cohorts, measure intervention response and retained margin, assess relevant subgroup behavior, monitor schema/freshness/calibration and govern model selection. Load only trusted local artifacts because pickle is executable serialization.
