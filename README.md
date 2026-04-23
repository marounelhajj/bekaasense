# BekaaSense

**A machine-learning system for desertification risk assessment in Lebanon's Bekaa Valley.**

BekaaSense forecasts the **De Martonne Aridity Index** and the
**Standardized Precipitation Index (SPI)** at monthly resolution for four
monitoring stations — **Ammik, Doures, Ras Baalbeck, Tal Amara** — classifies
future months into aridity zones, detects climate regime shifts, and
exposes the whole pipeline via a production-grade REST API and interactive
dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![Docker](https://img.shields.io/badge/Docker-ready-informational)
![Tests](https://img.shields.io/badge/tests-pytest-yellow)

---

## Key results (held-out test set, 2023–present)

| Model | Task | R² | RMSE | Coverage 90% | F1 (weighted) |
|---|---|---|---|---|---|
| LinearTrend | Regression (baseline) | −0.047 | 20.64 | — | — |
| SARIMA | Regression (baseline) | 0.837 | 8.15 | — | — |
| Rule | Classification (baseline) | — | — | — | 0.816 |
| **Random Forest** | **Regression** | **0.940** | **4.95** | **93.8%** | — |
| **XGBoost** | **Regression** | **0.928** | **5.42** | **94.4%** | — |
| **XGBoostClassifier** | **Classification** | — | — | — | **0.979** |

Prediction intervals use **split conformal prediction** with a finite-sample correction, guaranteeing ≥ 90% coverage on exchangeable test data.

---

## Quick start

### Option 1 — Docker (recommended)
```bash
cp .env.example .env
docker compose up --build
# Then open http://localhost:8000
```
The first run bakes the image (≈ 3 min). Afterwards it starts in seconds.

### Option 2 — local Python
```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_synthetic.py  # or: python scripts/build_canonical.py
python -m model_engine.train
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```
Then open <http://localhost:8000>.

### One-shot Makefile
```bash
make install      # pip install -r requirements.txt
make data         # generate synthetic dataset
make train        # train all 6 models (RF + XGB + classifier + 3 baselines)
make test         # pytest
make serve        # runserver :8000
make up           # docker compose up --build
```

---

## What the system does

1. **Ingests** heterogeneous station files (binary XLS, XML-SpreadsheetML,
   multi-sensor deployments for Tal Amara) into a canonical monthly schema.
2. **Imputes** missing values using station-month climatology (avoids the
   trap of rolling windows that leak future information) and flags every
   imputed cell so the UI can surface it.
3. **Engineers features** — lag-1/2/3 De Martonne, rolling precipitation
   (3/6/12 month), temperature rolls, SPI-3/6/12, cyclical month encoding,
   station one-hot — all with strict shift-1 leakage control. The nowcast
   feature set additionally includes the current-month De Martonne value,
   giving the zone classifier direct access to the value its thresholds are
   based on.
4. **Trains six models** on the same temporal split
   (train: 2015–2021, val: 2022, test: 2023+):

   | Model | Type | Task |
   |---|---|---|
   | Linear trend | Non-AI baseline | Regression |
   | SARIMA(1,0,1)(1,1,1,12) | Non-AI baseline | Regression |
   | Climatological rule | Non-AI baseline | Classification |
   | Random Forest (400 trees) | ML | Regression |
   | XGBoost (early stopping on val) | ML | Regression |
   | XGBoost Classifier (SMOTE-Tomek) | ML | Classification |

5. **Evaluates** every model on RMSE / MAE / R² / bias (regression) and
   weighted + macro F1 + per-class precision/recall/F1 + confusion matrix
   (classification). Conformal prediction intervals are computed for both
   ML regressors with a finite-sample coverage guarantee. A model health
   check flags any model that falls below quality thresholds.
6. **Explains** predictions with SHAP — global importance and per-row
   attributions, exposed on the dashboard and via `/api/explain/`.
7. **Serves** forecasts via a REST API and renders a Chart.js dashboard
   with: historical series + 24-month projection + 90% confidence band,
   aridity-zone decision guide, model explainer cards, AI-vs-baseline
   comparison charts, actual-vs-predicted scatter plot, 6 scientific
   conclusions, SHAP waterfall, classifier zone probabilities, and a
   crop-viability traffic light.

---

## Repository layout

```
bekaasense/
├── bekaasense/              Django project (settings, urls, wsgi, asgi)
├── data_ingestion/          App 1 — parsing, cleaning, feature engineering
│   ├── indices.py           De Martonne + SPI (McKee et al. 1993)
│   ├── loaders.py           Heterogeneous raw-file loader
│   ├── cleaners.py          Imputation + outlier flagging
│   └── features.py          Lag/roll/SPI features; nowcast set includes de_martonne
├── model_engine/            App 2 — ML models + training
│   ├── baselines.py         3 non-AI baselines (linear, SARIMA, rule)
│   ├── ml_models.py         RF + XGBoost regressors; XGBoostClassifier (SMOTE-Tomek)
│   ├── explainability.py    SHAP global + per-row
│   ├── evaluate.py          Metrics + residual stratification
│   ├── train.py             End-to-end orchestrator; runs RF+XGB classifiers, picks best
│   └── inference.py         Online prediction + crop-viability signal
├── api/                     App 3 — DRF endpoints
│   ├── views.py             All API views (see endpoint table below)
│   ├── serializers.py
│   └── urls.py
├── dashboard/               App 4 — UI
│   ├── templates/dashboard/index.html
│   └── static/dashboard/{css,js}/
├── scripts/                 CLI utilities
│   ├── generate_synthetic.py    calibrated demo data
│   ├── build_canonical.py       real-data ingestion
│   └── show_leaderboard.py
├── tests/                   pytest — indices, features (leakage), baselines, API
├── data/
│   ├── raw/                 (gitignored) drop your XLS/CSV files here
│   └── processed/           bekaa_valley_clean.csv
├── results/
│   └── metrics/             leaderboard.{csv,json}, classifier_report.json,
│                            model_health.json, shap_importance.csv,
│                            residuals_by_{year,station}.csv, test_predictions.csv
├── model_engine/artifacts/  Persisted models (joblib) + feature_sets.json
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── pytest.ini
└── readme_correction.md     Rubric-to-file map (EECE 490/690)
```

---

## API

All endpoints are JSON. Full schemas in `api/serializers.py`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health/` | GET | Liveness probe |
| `/api/stations/` | GET | List stations + latest observation + viability signal |
| `/api/predict/` | POST | Forecast N months with calibrated 90% interval |
| `/api/classify/` | POST | Aridity zone for a given (station, year, month) |
| `/api/trend/` | GET | Historical series + 24-month forecast for one station |
| `/api/explain/` | POST | SHAP attribution for latest feature row |
| `/api/leaderboard/` | GET | Model comparison table |
| `/api/scoring/` | GET | Full scoring report: per-class metrics, confusion matrix, health |
| `/api/latest_zone/` | GET | Classifier prediction + probabilities for the latest month |
| `/api/test_predictions/` | GET | Held-out test set predictions (for comparison plots) |

### Example

```bash
curl -X POST http://localhost:8000/api/predict/ \
     -H "Content-Type: application/json" \
     -d '{"station": "Ammik", "horizon_months": 12, "alpha": 0.1}'
```

Response:

```json
{
  "station": "Ammik",
  "horizon_months": 12,
  "nominal_coverage": 0.9,
  "forecast": [
    {"year": 2026, "month": 5, "horizon": 1,
     "de_martonne_pred": 4.12, "lower": 2.04, "upper": 6.19,
     "aridity_zone": "Hyper-arid"},
    ...
  ]
}
```

---

## Responsible ML

BekaaSense addresses **four** responsible-ML dimensions:

- **Explainability (RM1).** SHAP global importance + per-row attributions
  are persisted to `results/metrics/shap_importance.csv` and surfaced via
  `/api/explain/` and the dashboard SHAP waterfall chart.
- **Bias (RM2).** Station-level residual stratification in
  `results/metrics/residuals_by_station.csv`. Imputed cells carry an
  `imputed_<col>` flag so no downstream consumer silently treats imputed
  data as observed.
- **Data integrity / privacy (RM3).** No personally identifiable data is
  processed. Raw station files are `.gitignore`d to respect data-provider
  licensing.
- **Robustness (RM4).** Prediction intervals use **split conformal
  prediction** with a finite-sample correction (`ceil((n+1)(1−α))/n`),
  guaranteeing ≥ 90% empirical coverage on exchangeable test data
  (RF: 93.8%, XGB: 94.4% actual coverage). Residual analysis stratified
  by year detects model degradation over time. A `model_health.json`
  report flags any model failing quality thresholds after training.

---

## Honest limitations

- The ~10-year training record limits long-horizon forecast reliability.
  Prediction intervals widen with horizon by design.
- Four monitoring stations provide limited spatial resolution; no claim
  is made for unmonitored locations.
- The De Martonne formulation does not include wind, solar radiation, or
  actual evapotranspiration. SPEI is a future enhancement.
- Mann–Kendall trends are **not yet statistically significant** on any
  station (p > 0.05) — 20–30 years of data would be needed.

---

## License & citation

Educational project for **EECE 490/690 — Introduction to Machine
Learning, Spring 2025–2026, American University of Beirut.**

For a criterion-by-criterion mapping of the course rubric to files and
line numbers, see [`readme_correction.md`](readme_correction.md).