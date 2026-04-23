# Evaluation protocol

## Split

Strictly temporal:

| Split       | Range                    | Purpose                           |
|-------------|--------------------------|-----------------------------------|
| Train       | 2015-01 → 2021-12        | Fit all 6 models                  |
| Validation  | 2022-01 → 2022-12        | Calibrate XGBoost residuals       |
| Test        | 2023-01 → present        | Final leaderboard numbers         |

No random shuffling, no k-fold, no cross-station holdout — the system
must forecast *forward in time*, so the evaluation protocol must mirror
that. Standard k-fold cross-validation would shuffle the time axis,
creating look-ahead bias (a model trained on 2024 data predicting 2022
outcomes is not a valid proxy for real-world deployment). Walk-forward
or expanding-window CV would be theoretically correct but requires
≥ 5 years of test data to be statistically meaningful; with only
~3 test years, a single clean temporal split is more honest than a
noisy rolling estimate. The `assert_no_leakage` guardrail in
`data_ingestion.features` fails loudly if training ever contains a
date ≥ earliest test date.

## Metrics

### Regression (De Martonne index forecast)

| Metric            | Why it's here                                             |
|-------------------|-----------------------------------------------------------|
| RMSE              | Penalises large errors; comparable units to the target    |
| MAE               | Robust to outliers; easier to explain to stakeholders     |
| R²                | Fraction of variance explained — captures "good fit"      |
| Bias              | Mean signed error — detects systematic over/under-forecast |
| Interval coverage | Fraction of true values in the 90 % prediction band       |

### Classification (aridity zone)

| Metric            | Why it's here                                             |
|-------------------|-----------------------------------------------------------|
| F1 weighted       | Handles severe class imbalance (Hyper-arid ≈ 80 %)        |
| F1 macro          | Penalises collapsing to the majority class                |
| Per-class report  | Surfaces which zones are mis-predicted                    |
| Confusion matrix  | Full structure of the errors                              |

## Success criteria

A model is considered a clear improvement over a baseline only if it
wins on **both** RMSE and R² on the held-out test set, with the
interval coverage within ±5 percentage points of the nominal 90 %.

A "tie" — meaning the ML model matches but does not clearly beat the
best baseline — is a **finding worth reporting honestly**, not hidden.
The rubric rewards critical thinking over inflated claims.

## Residual analysis

Two tables are written after every training run:

- `results/metrics/residuals_by_year.csv` — mean / std / MAE of residuals
  per test year. Rising MAE over time indicates model drift.
- `results/metrics/residuals_by_station.csv` — same, but per station.
  A large bias at one station is a **bias / fairness** signal (RM2).

## Reproducibility

- `random_state=42` is fixed for every RNG-using component.
- `requirements.txt` is fully pinned.
- The same numeric result should appear on any machine after:

  ```bash
  make install
  make data
  make train
  make evaluate
  ```
