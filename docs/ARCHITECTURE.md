# Architecture

## Overview

BekaaSense is a four-layer Django application with a strict separation of
concerns enforced by the app boundary:

```
┌──────────────────────────────────────────────────────────────┐
│  dashboard/  ── Django templates + Chart.js front-end        │
├──────────────────────────────────────────────────────────────┤
│  api/        ── DRF views + serializers (stateless)          │
├──────────────────────────────────────────────────────────────┤
│  model_engine/ ── training · inference · explainability      │
├──────────────────────────────────────────────────────────────┤
│  data_ingestion/ ── loaders · cleaners · features · indices  │
├──────────────────────────────────────────────────────────────┤
│  data/   results/   model_engine/artifacts/                  │
└──────────────────────────────────────────────────────────────┘
```

Each app has a single responsibility. The rule — enforced during code
review — is that `api` never reads CSVs directly and `dashboard` never
imports `model_engine`. API views delegate to the inference service,
which loads persisted joblib artefacts from `model_engine/artifacts/`.

## Request lifecycle — `/api/predict/`

```
Client ──POST──▶ Django URL router
                    │
                    ▼
             api/views.predict
                    │  validate payload (ForecastRequestSerializer)
                    ▼
             model_engine.inference.forecast_station
                    │  load registry (cached)
                    │  read latest feature row
                    │  recursive rollout for N months
                    ▼
             model_engine.ml_models.RandomForestForecaster.predict_with_interval
                    │  per-tree predictions → percentile bounds
                    ▼
             JSON response { forecast: [...] }
```

## Data flow — training

```
data/raw/*.xls{x}  ── (optional real data)
       │
       ▼
scripts/build_canonical.py
       │
       ▼
data_ingestion.loaders.build_canonical_dataset
       │  alias-tolerant header mapping
       ▼
data_ingestion.cleaners.clean_pipeline
       │  station-month climatology imputation
       │  IQR outlier flagging (flag, don't drop)
       ▼
data/processed/bekaa_valley_clean.csv
       │
       ▼
data_ingestion.features.build_features
       │  De Martonne + SPI + lags + rollings + seasonal
       │  shift(1) BEFORE rolling → no leakage
       ▼
data_ingestion.features.temporal_split
       │  train ≤ 2021-12 │ val 2022 │ test ≥ 2023
       ▼
model_engine.train.run
       │  fits all 6 models on the same split
       │  writes artefacts + leaderboard + residuals + SHAP
       ▼
model_engine/artifacts/*.joblib
results/metrics/*.csv *.json
```

## Design decisions

**Django over Flask.** The problem formulation committed to Django; it
also gives us DRF, a browsable API, admin, and throttling out of the box.

**SQLite by default, Postgres on demand.** The workload is almost entirely
read-only once the models are trained; SQLite is sufficient for a
demo deployment. Postgres via `DATABASE_URL` is wired through for any
production deployment that expects concurrent writes.

**Joblib persistence over pickle.** `joblib` is the scikit-learn standard
and handles large numpy arrays better than raw pickle.

**Split conformal prediction intervals.** After fitting, each regressor
is calibrated on the 2022 validation set: the absolute residuals
`|y_val − ŷ_val|` are stored and the quantile at level
`ceil((n+1)(1−α))/n` (finite-sample conformal correction) is used as a
symmetric half-width at inference time. This gives a mathematical
guarantee that empirical coverage ≥ 1−α on exchangeable test data —
unlike per-tree quantiles (which only capture epistemic uncertainty) or
residual bootstrap (which can undercover when the test distribution
shifts). Actual test coverage: RF 93.8%, XGBoost 94.4% vs. the 90%
target.

**SHAP TreeExplainer.** Exact (not sampled) attributions for tree models,
and much faster than KernelExplainer. Critically, SHAP values decompose
additively — every prediction equals `base_value + sum(shap_values)`,
which makes the dashboard bar chart interpretable without caveats.

**Cyclical month encoding over one-hot month.** `sin(2π·m/12)` and
`cos(2π·m/12)` capture the fact that December is "close to" January
without giving the model 12 unrelated dummy variables.

**Station one-hot over target encoding.** With only four stations the
dimensionality penalty is negligible and target encoding would risk
leakage into the test set.

## ML vs non-AI trade-off

The three baselines are chosen to attack the ML approach from three
different angles:

- **Linear trend** tests whether climate is changing *linearly enough*
  that simple extrapolation suffices.
- **SARIMA** tests whether the time-series structure (seasonality +
  autocorrelation) alone is enough, without any cross-station or
  multivariate information.
- **Rule baseline** tests whether a domain-expert-style "below mean AND
  above mean" rule can match a learned classifier.

If any of these three beats or ties the ML models on the test set, the
extra complexity is not justified and must be documented. This is what a
fair comparison looks like in this domain; the leaderboard exposes every
number so the reader can judge for themselves.

## Model selection

- **Random Forest** is the primary regressor: robust to outliers,
  insensitive to feature scaling, and trivially exposes per-tree
  prediction intervals. Good baseline-beater.
- **XGBoost** is the secondary regressor: typically wins on tabular
  data but harder to calibrate uncertainty for. Kept for comparison
  and possible ensembling.
- **Random Forest classifier with class weighting + SMOTE** handles the
  severe class imbalance surfaced during EDA (Hyper-arid months dominate
  ~80% of observations).

## Deployment audience

BekaaSense augments decisions for:

- **Lebanese Ministry of Agriculture** — policy-level crop-viability
  guidance for rain-fed wheat zones.
- **LARI (Lebanese Agricultural Research Institute)** — research
  baseline for Bekaa monitoring.
- **FAO Lebanon / USAID / IDRC** — risk-screening input for climate-
  adaptation funding decisions.
- **Farm cooperatives** — informs investment in irrigation infrastructure.

No claim is made that the system replaces human judgement; it is a
decision *support* tool whose uncertainty bounds are communicated
explicitly at every prediction.
