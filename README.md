# BekaaSense — Desertification Risk Intelligence for Lebanon's Bekaa Valley

<div align="center">

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Azure-0078D4?style=for-the-badge&logo=microsoft-azure)](https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/marounelhajj/bekaasense)
[![EDA Notebook](https://img.shields.io/badge/EDA-Google%20Colab-F9AB00?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/drive/19guFdh_mkdBxEdAC_WST_Fp-qpq9jsLc)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-092E20?style=for-the-badge&logo=django)](https://djangoproject.com)
[![Tests](https://img.shields.io/badge/Tests-17%20passing-22c55e?style=for-the-badge)](tests/)

*A production-grade machine-learning system for aridity forecasting and desertification risk assessment
at four LARI monitoring stations in Lebanon's Bekaa Valley.*

**[→ Open the Live Dashboard](https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net)**  
**[→ Explore the EDA Notebook](https://colab.research.google.com/drive/19guFdh_mkdBxEdAC_WST_Fp-qpq9jsLc)**

</div>

---

## Key Results — held-out test set (2023–present, 144 samples)

| Model | Task | RMSE | R² | F1 (weighted) | Coverage |
|---|---|---|---|---|---|
| Linear Trend | Regression | 20.64 | −0.047 | — | — |
| SARIMA | Regression | 8.15 | 0.837 | — | — |
| Rule Baseline | Classification | — | — | 0.816 | — |
| **Random Forest** | **Regression** | **4.95** | **0.940** | — | **93.8%** |
| XGBoost | Regression | 5.42 | 0.928 | — | 94.4% |
| XGBoost Classifier | Classification | — | — | **0.979** | — |

> Random Forest reduces RMSE by **39%** over the best non-AI baseline (SARIMA). Both ML regressors exceed the 90% conformal prediction interval coverage target.

---

## Abstract

Lebanon's Bekaa Valley is the country's agricultural backbone, generating an estimated **USD 700 million
per year** and supplying roughly 40% of national food output. This work presents **BekaaSense**, an
end-to-end machine-learning system that forecasts the **De Martonne Aridity Index** at monthly
resolution up to 24 months ahead at four Lebanese Agricultural Research Institute (LARI) monitoring
stations: Ammik, Doures, Ras Baalbeck, and Tal Amara.

Six models — three classical baselines and three machine-learning models — are trained and evaluated
on a strict temporal holdout to ensure fair, unbiased comparison. The best regressor, a calibrated
**Random Forest** with split conformal prediction intervals, achieves **R² = 0.940** and **RMSE = 4.95
DM units** on the held-out test set, reducing forecast error by **39% over the strongest statistical
baseline** (SARIMA, RMSE = 8.15). A companion **XGBoost Classifier** identifies aridity zones with
**F1 weighted = 0.979** (140 out of 144 correct), and both regressors exceed the 90% nominal
prediction interval coverage target (RF: **93.8%**, XGB: **94.4%**).

The system is deployed live on **Microsoft Azure**, exposes a full REST API, and renders an interactive
dashboard with forecasts, uncertainty bands, SHAP explanations, an agricultural decision guide, and
scientific conclusions — designed to support decision-making by the Lebanese Ministry of Agriculture,
LARI, FAO Lebanon, and farm cooperatives.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Results](#3-results)
4. [Conclusions](#4-conclusions)
5. [Honest Limitations](#5-honest-limitations)
6. [Quick Start](#6-quick-start)
7. [CI/CD Pipeline & Cloud Deployment](#7-cicd-pipeline--cloud-deployment)
8. [API Reference](#8-api-reference)
9. [System Capabilities](#9-system-capabilities)

---

## 1. Introduction

### The Problem

Lebanon's agricultural economy is critically exposed to climate variability. The Bekaa Valley — a
high-altitude inland plateau bounded by the Lebanon and Anti-Lebanon ranges — receives 80–90% of its
precipitation in winter and is almost entirely dry in summer. Rain-fed wheat, the dominant cereal
crop, requires a De Martonne Aridity Index (DM) of at least **20** to remain economically viable.
When DM falls below 20, yields collapse and irrigation costs become prohibitive for smallholder farmers.

Climate trends in the Eastern Mediterranean are moving in the wrong direction. Long-term datasets show
Lebanon warming at roughly **0.4–0.5 °C per decade** with declining precipitation, compressing the
window in which rain-fed agriculture is viable. Against the backdrop of Lebanon's ongoing economic
crisis (GDP contracted by more than 40% between 2019 and 2021), the erosion of Bekaa agriculture
could prove catastrophic for food security.

**The core question BekaaSense is built to answer:**
> *Is rain-fed wheat still viable at this station, in this month, and for how long into the future?*

### Why Machine Learning

Three non-AI baselines were built specifically to answer whether simpler approaches suffice:

| Approach | Core limitation | Test result |
|---|---|---|
| Linear trend | Cannot capture seasonal cycles | R² = −0.047 (worse than predicting the mean) |
| SARIMA | Cannot learn nonlinear multi-variable interactions | RMSE = 8.15 DM units |
| Expert rule ("drier if below-average rainfall AND above-average temp") | Binary only; no continuous probability | F1 = 0.816 |

The 39% RMSE reduction of Random Forest over SARIMA represents a practically meaningful gain — the
difference between a ±5 DM and ±8 DM uncertainty band can determine whether a forecast falls above
or below the critical DM = 20 wheat viability threshold.

### Intended Users

| Stakeholder | Use case |
|---|---|
| Lebanese Ministry of Agriculture | Crop-viability guidance; irrigation subsidy allocation |
| LARI | Station-level research baseline; annual aridity reports |
| FAO Lebanon / USAID / IDRC | Risk-screening for climate-adaptation funding decisions |
| Farm cooperatives | Investment planning for irrigation infrastructure |

---

## 2. Methodology

### Data

**Source:** Lebanese Agricultural Research Institute (LARI), four Bekaa Valley stations.  
**Coverage:** Monthly observations, January 2015 – present (~11 years, 528+ station-months).

#### Raw Data Files

The raw data is provided by LARI as Microsoft Excel files, one per station. Tal Amara is split across multiple sub-files (rain gauge and temperature loggers maintained separately):

```
Climatic data/
├── Ammik.xls              ← monthly temp + precip, Ammik station
├── Doures.xls             ← monthly temp + precip, Doures station
├── Ras Baalbeck.xls       ← monthly temp + precip, Ras Baalbeck station
└── Tal Amara/
    ├── Rain.xls           ← precipitation logger
    ├── TA1.xls            ← temperature sensor 1
    ├── TA2.xls            ← temperature sensor 2
    └── TA3.xls            ← temperature sensor 3
```

These files are loaded by `data_ingestion/loaders.py`, merged, cleaned in `data_ingestion/cleaners.py`, and the processed output is written to `data/processed/bekaa_valley_clean.csv`. The raw `.xls` files are not committed to the repository as they are proprietary LARI data; contact [lari.gov.lb](http://www.lari.gov.lb/) to obtain them.

| Station | Elevation (m) | Latitude | Longitude |
|---|---|---|---|
| Ammik | ~980 | 34.09°N | 35.95°E |
| Doures | ~920 | 33.87°N | 35.90°E |
| Ras Baalbeck | ~1,190 | 34.26°N | 36.41°E |
| Tal Amara | ~870 | 33.84°N | 35.98°E |

**Target variable — De Martonne Aridity Index:**

$$I_m = \frac{12 \times P_m}{T_m + 10}$$

where $P_m$ is monthly precipitation (mm) and $T_m$ is monthly mean temperature (°C).

**Aridity zone thresholds:**

| Zone | DM Range | Agricultural significance |
|---|---|---|
| 🔴 Hyper-arid | DM < 5 | Extreme drought; irrigation critical |
| 🟠 Arid | 5 ≤ DM < 10 | Severe deficit; rain-fed agriculture unviable |
| 🟡 Semi-arid | 10 ≤ DM < 20 | Water stress; rain-fed wheat at risk |
| 🟢 Sub-humid | 20 ≤ DM < 30 | Viable with monitoring |
| 🔵 Humid | DM ≥ 30 | Excellent conditions |

**Missing data:** Station-month climatological imputation (median of the same calendar month across
years). Every imputed cell is flagged with an `imputed_<col>` boolean so consumers can distinguish
observed from imputed values.

### Feature Engineering

All features enforce **strict temporal leakage control**: every rolling window uses `shift(1)` before
aggregation so no row at time *t* contains information from time *t* itself.

**Forecast features (15) — used by regression models:**

| Feature | Description |
|---|---|
| `month_sin`, `month_cos` | Cyclical month encoding (avoids Dec/Jan discontinuity) |
| `dm_lag1/2/3` | Lagged De Martonne (1–3 months) |
| `precip_lag1`, `temp_lag1` | Lagged precipitation and temperature |
| `precip_roll3/6/12` | Rolling mean precipitation (3/6/12 months) |
| `dm_roll3/6` | Rolling mean De Martonne |
| `temp_roll3` | Rolling mean temperature |
| `spi3_lag1`, `spi6_lag1` | Lagged Standardized Precipitation Index |

**Nowcast features (21) — used by classifier, adds current observations:**  
All forecast features + `precip_sum`, `temp_avg/max/min`, `de_martonne`, `spi3/6/12`, station one-hot.

> Including `de_martonne` in the nowcast set is the single most impactful design decision: without it,
> the classifier achieved F1 macro = 0.871 (forced to reverse-engineer `12P/(T+10)` from raw inputs).
> With it: **F1 macro = 0.974**.

**Temporal split:**

```
Training (2015–2021)   Validation (2022)   Test — held out (2023–present)
     332 rows               48 rows                  144 rows
```

### Models

| Model | Type | Key parameters |
|---|---|---|
| LinearTrend | Non-AI baseline | OLS of annual DM on year, per station |
| SARIMA(1,0,1)(1,1,1)₁₂ | Non-AI baseline | Seasonal ARIMA |
| Rule Baseline | Non-AI baseline | `precip_roll3 < mean AND temp_avg > mean` |
| **Random Forest** | ML regressor | 400 trees, max_depth=10, conformal intervals |
| **XGBoost** | ML regressor | Early stopping on val set (stopped at iter 49/600) |
| **XGBoost Classifier** | ML classifier | SMOTE-Tomek, balanced weights |

**Prediction intervals** use **split conformal prediction** with finite-sample correction:

$$q = Q_{|r|}\!\left(\frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

This provides a mathematical guarantee that empirical coverage ≥ (1−α) on exchangeable test data.

**Overfitting audit:** Before adding early stopping, XGBoost had train R² = 0.9999 — catastrophic
overfitting. After: train R² = 0.94, test R² = 0.93, gap = 0.014. Random Forest train–test gap ≈ 0.

### Evaluation

- **Regression:** RMSE, MAE, R², Bias, Interval Coverage
- **Classification:** F1 weighted, F1 macro, per-class precision/recall/F1, confusion matrix
- **Robustness:** Residuals stratified by year (drift detection) and station (bias detection)
- **Explainability:** SHAP TreeExplainer — global importance + per-prediction attributions

---

## 3. Results

### Regression — De Martonne Forecasting

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Bias | Coverage 90% |
|---|---|---|---|---|---|
| LinearTrend | 20.64 | 17.84 | −0.047 | 5.03 | — |
| SARIMA | 8.15 | 4.38 | 0.837 | 1.92 | — |
| **Random Forest** | **4.95** | **3.17** | **0.940** | 0.92 | **93.8%** |
| XGBoost | 5.42 | 4.00 | 0.928 | 1.12 | 94.4% |

- **LinearTrend fails outright** (R² = −0.047) — worse than predicting the historical mean
- **SARIMA captures 83.7% of variability** — the seasonal floor any model must clear
- **Random Forest cuts RMSE by 39%** over SARIMA — nonlinear multi-variable interactions drive the gain
- **RF and XGBoost converge at R² ≈ 0.93–0.94** — strong evidence this is the information ceiling of the 11-year dataset

### Classification — Aridity Zone Identification

| Zone | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Humid | 1.000 | 1.000 | 1.000 | 44 |
| Sub-humid | 1.000 | 0.938 | 0.968 | 16 |
| Semi-arid | 0.952 | 1.000 | 0.976 | 20 |
| Arid | 0.952 | 0.952 | 0.952 | 21 |
| Hyper-arid | 0.977 | 0.977 | 0.977 | 43 |
| **Macro avg** | **0.976** | **0.973** | **0.974** | **144** |

**140 out of 144 correct.** The 4 remaining misclassifications all occur within 1 DM unit of a
zone boundary — genuinely ambiguous cases where even a human expert with raw data would face
the same difficulty.

**Before → After adding de_martonne feature:**

| Zone | F1 Before | F1 After | Improvement |
|---|---|---|---|
| Sub-humid | 0.667 | 0.968 | **+0.301** |
| Semi-arid | 0.789 | 0.976 | **+0.187** |
| Macro avg | 0.871 | **0.974** | **+0.103** |

### Prediction Interval Coverage

| Model | Method | Target | Actual | Status |
|---|---|---|---|---|
| Random Forest | Split conformal | 90% | 93.8% | ✅ |
| XGBoost | Split conformal | 90% | 94.4% | ✅ |

Before implementing conformal prediction: RF coverage = 81.9%, XGB = 87.5% — both below target.

### SHAP Feature Importance (Global, Random Forest)

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | `month_cos` | 10.82 | Seasonal timing dominates |
| 2 | `dm_lag1` | 8.19 | Short-term aridity persistence |
| 3 | `precip_lag1` | 1.89 | Prior-month rainfall |
| 4 | `precip_roll12` | 1.42 | Annual moisture accumulation |
| 5 | `spi6_lag1` | 1.23 | 6-month drought context |

---

## 4. Conclusions

### Key Findings

1. **Linear extrapolation is actively harmful** (R² = −0.047) — monthly Bekaa aridity is a
   seasonal phenomenon, not a trend. Simple extrapolation misleads more than it informs.

2. **Seasonality explains 84% of variability** — the Mediterranean winter-rainfall / summer-drought
   cycle is so dominant that past values alone yield solid baseline performance (SARIMA R² = 0.837).

3. **Machine learning reduces forecast error by 39%** — nonlinear interactions between
   precipitation lags, temperature anomalies, and multi-scale drought indices (SPI-3, SPI-6)
   are the source of this gain, and cannot be captured by SARIMA.

4. **Convergence of RF and XGBoost at R² ≈ 0.93–0.94** suggests this is the information ceiling
   of the current 11-year, 4-station dataset. Exceeding it would require longer records or
   additional covariates (satellite imagery, reanalysis data).

5. **Zone classification is near-perfect (F1 = 0.979)** once the computed De Martonne value
   is included — confirming the bottleneck was a feature engineering gap, not a modelling limitation.

6. **Prediction intervals are reliable and conservative** (93–94% actual vs 90% nominal) —
   the system errs toward wider intervals, the correct behavior for agricultural risk management.

### Limitations

See [Section 5 — Honest Limitations](#5-honest-limitations) for a full discussion.

### Future Work

| Priority | Direction |
|---|---|
| 🔴 High | Extend record with ERA5 reanalysis back to 1980+ |
| 🔴 High | Add SPEI as a second target — more complete than De Martonne |
| 🟡 Medium | Satellite covariates (NDVI, soil moisture from Sentinel/MODIS) |
| 🟡 Medium | Temporal Fusion Transformer for improved long-horizon forecasting |
| 🟢 Low | Real-time data pipeline from LARI when API becomes available |

---

## 5. Honest Limitations

This section documents every known limitation of the system. These are not excuses — they are facts that any user making a real decision should be aware of.

### Data Limitations

**Small dataset.** The system is trained on approximately 528 station-months across 4 stations and 11 years. This is a small dataset by ML standards. Complex climate patterns cannot be reliably learned from this volume of data. The R² ceiling of ~0.94 is almost certainly the information limit of the dataset, not a modelling limit — adding more powerful models would not meaningfully improve performance without more data.

**Single source.** All data comes from LARI. If LARI sensors have systematic calibration errors or biases, the models inherit them. There is no independent validation dataset from a second source.

**Sparse station network.** Four stations cover an entire valley. The spatial resolution is coarse. Predictions are valid only at the exact location of each station. A farm 10 km from the nearest station may have materially different conditions that this system cannot detect or forecast.

**Manual data ingestion.** LARI has no public API. Data is uploaded manually. The system does not update automatically when new station readings become available. In production, this would require a scheduled ingestion pipeline that does not currently exist.

### Model Limitations

**Trend significance.** Mann–Kendall trend tests across all four stations return p > 0.05. The apparent trends in the data — Ammik wetting slightly, the other three drying — are directionally consistent with regional Eastern Mediterranean climatology, but are **not statistically significant** at conventional levels given only 11 years. The system should not be used to make claims about long-term climate trends.

**Long-horizon reliability.** Prediction intervals at 18–24 months are very wide (typically ±12–15 DM units). At these horizons the interval often spans multiple aridity zones, making the zone forecast effectively uninformative. Operationally useful forecasts are those at 1–6 months. The 24-month maximum is presented for completeness, not as an operational recommendation.

**Index simplicity.** De Martonne uses only two variables: monthly precipitation and mean temperature. It ignores wind speed, solar radiation, relative humidity, and actual evapotranspiration. The SPEI (Standardized Precipitation-Evapotranspiration Index) is a more physically complete aridity indicator, but it requires longer homogeneous records than the current dataset provides.

**Distribution shift.** The models are trained on 2015–2025 climate. If climate conditions in 2030+ fall outside the range seen during training, model errors will increase. The widening prediction intervals with horizon provide a partial mitigation, but they do not correct for structural shifts in the climate regime.

**No walk-forward cross-validation.** With only 11 years of data, rolling cross-validation windows would each be too small (under 100 rows) to train reliable models, making the CV estimates themselves unreliable. A single temporal split (train 2015–2021, val 2022, test 2023+) is the most honest choice given the data size, but it means performance estimates are based on a single test window rather than multiple independent windows.

### What the System Cannot Do

| Capability | Status | Why |
|---|---|---|
| Real-time data ingestion | Not implemented | LARI has no public API |
| Spatial interpolation between stations | Not implemented | Would require a denser sensor network and geostatistical methods |
| Day-level or week-level forecasting | Not implemented | Data is monthly-aggregate only |
| Forecasts beyond 24 months | Not supported | Interval too wide to be informative; outside validated range |
| Claims about climate trends | Not supported | Mann–Kendall p > 0.05 on all stations |
| Generalization to stations outside the 4 monitored | Not supported | Model has no spatial generalization |
| SPEI or PET-based aridity | Not implemented | Insufficient data for homogeneous long-record estimation |

---

## 6. Quick Start

### Docker (recommended)
```bash
git clone https://github.com/marounelhajj/bekaasense.git
cd bekaasense
docker compose up --build
# Open http://localhost:8000
```

### Local Python
```bash
git clone https://github.com/marounelhajj/bekaasense.git
cd bekaasense
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synthetic.py
python -m model_engine.train
python manage.py migrate && python manage.py runserver
```

### Makefile
```bash
make install && make data && make train && make test && make serve
```

---

## 7. CI/CD Pipeline & Cloud Deployment

Every push to the `main` branch triggers two automated GitHub Actions workflows before the live site is updated.

### Workflow 1 — CI (`ci.yml`)

Runs on every push and pull request. Steps in order:

```
1. Checkout code
2. Set up Python 3.11
3. Install all dependencies (pip install -r requirements.txt)
4. Lint — python -m compileall (catches syntax errors in all apps)
5. Generate synthetic dataset (scripts/generate_synthetic.py)
6. Train all 6 models on synthetic data (python -m model_engine.train)
7. Run 17 pytest tests across 4 test files
8. Upload leaderboard.json as a downloadable artifact
```

The 17 tests cover:
- **`test_indices.py`** — De Martonne formula correctness, SPI behaviour, zone threshold classification
- **`test_features.py`** — Temporal split ordering, leakage guardrail (fails if a random split is used), lag feature correctness, forecast feature set does not contain current-step values
- **`test_baselines.py`** — LinearTrend, SARIMA, and Rule baseline fit and produce finite predictions
- **`test_api.py`** — `/health/` returns 200, `/api/stations/` returns 503 (not a crash) without data, `/api/predict/` returns 400 for an invalid station

### Workflow 2 — Deploy (`main_bekaasense.yml`)

Runs only after the CI workflow passes. Steps:

```
1. Build the app (install dependencies in a clean virtualenv)
2. Upload build artifact
3. Login to Azure using stored GitHub secrets
   (client ID, tenant ID, subscription ID — never in the codebase)
4. Deploy to Azure App Service — Production slot, app name "Bekaasense"
```

### End-to-end flow

```
git push origin main
        │
        ▼
GitHub Actions: CI workflow
  ├── lint → synthetic data → train → 17 tests
  └── all pass? ──No──▶ pipeline stops, Azure is NOT updated
                │
               Yes
                ▼
GitHub Actions: Deploy workflow
  └── build → Azure login → deploy to Production slot
                │
                ▼
Live site updated (~3–5 min total from push)
https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net
```

The Azure credentials are stored as GitHub repository secrets and are never present in the codebase or commit history.

---

## 8. API Reference

Base URL: `https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net`

| Endpoint | Method | Description |
|---|---|---|
| `/api/stations/` | GET | All 4 stations with latest observation and crop viability |
| `/api/predict/` | POST | Forecast N months with calibrated 90% interval |
| `/api/classify/` | POST | Aridity zone for (station, year, month) |
| `/api/trend/` | GET | Historical series + 24-month forecast |
| `/api/explain/` | POST | SHAP attribution for the latest feature row |
| `/api/leaderboard/` | GET | Full model comparison table |
| `/api/scoring/` | GET | Per-class metrics, confusion matrix, model health |
| `/api/latest_zone/` | GET | Classifier prediction + probabilities for the latest month |
| `/api/test_predictions/` | GET | Held-out test set predictions |

**Example:**
```bash
curl -X POST https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net/api/predict/ \
     -H "Content-Type: application/json" \
     -d '{"station": "Ammik", "horizon_months": 12, "alpha": 0.1}'
```

---

## 9. System Capabilities

### What is implemented

- **End-to-end ML pipeline** — data ingestion → cleaning → feature engineering → training → evaluation → serving, all reproducible with `make train`
- **6-model comparison** — 3 non-AI baselines (LinearTrend, SARIMA, Rule) + 3 ML models (RF regressor, XGB regressor, XGB classifier), all evaluated on the same held-out test set
- **Strict leakage control** — shift-1 before rolling windows; `assert_no_leakage` guardrail test in CI
- **Conformal prediction intervals** — mathematically guaranteed ≥ 90% coverage; finite-sample corrected
- **SHAP explainability** — global importance + per-prediction attributions via API and dashboard
- **Responsible ML** — imputation flags, per-station residual bias audit, year-stratified drift monitoring, model health check after every training run
- **Production deployment** — multi-stage Dockerfile (non-root, healthcheck), Azure App Service with CI/CD on push to main
- **Interactive dashboard** — Chart.js with forecast chart, 90% confidence band, SHAP waterfall, zone classifier, crop viability signal, model explainer cards, AI vs baseline comparison charts, scientific conclusions
- **REST API** — 10 endpoints, full JSON, DRF with input validation
- **Full test suite** — 17 passing tests covering indices, feature leakage, baselines, and API

### Responsible ML coverage (all four dimensions)

| Dimension | Implementation |
|---|---|
| RM1 — Explainability | SHAP TreeExplainer: global importance + per-prediction attributions via `/api/explain/` and dashboard bar chart |
| RM2 — Fairness / bias | Per-station residual analysis (spatial bias check): `/api/residuals/` + dashboard error analysis charts |
| RM3 — Privacy / leakage | Raw data excluded from repo (`.gitignore`); automated leakage guardrail test in CI that fails if shift-1 rule is violated |
| RM4 — Robustness / shift | Split conformal prediction intervals (mathematical ≥90% coverage guarantee); model health check after every training run; year-stratified residuals for temporal drift detection |

All four RM dimensions are addressed. The interval coverage metric (RM4) is a second robustness check beyond residual analysis.

### What is not implemented (acknowledged limitations)

- Real-time LARI data ingestion — currently requires manual CSV upload
- Spatial interpolation between stations — predictions are point-level only
- SPEI index — requires longer homogeneous temperature records; planned future work
- Walk-forward cross-validation — dataset is too short (11 years) for meaningful rolling CV windows; single temporal split used instead

---

## References

- De Martonne, E. (1926). Une nouvelle fonction climatologique: l'indice d'aridité. *La Météorologie*, 2, 449–458.
- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conf. on Applied Climatology*, AMS.
- Angelopoulos, A.N., Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Lundberg, S.M., Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- FAO. (2023). *Lebanon: Country Agro-Informatics Profile*. Food and Agriculture Organization.
- World Bank. (2021). *Lebanon Economic Monitor — Lebanon Sinking (To the Top Three)*. World Bank Group.

---

<div align="center">

**BekaaSense** · American University of Beirut · Spring 2025–2026

Built with Django · scikit-learn · XGBoost · SHAP · Chart.js · Deployed on Microsoft Azure

[Dashboard](https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net) ·
[API](https://bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net/api/stations/) ·
[Source](https://github.com/marounelhajj/bekaasense) ·
[EDA Notebook](https://colab.research.google.com/drive/19guFdh_mkdBxEdAC_WST_Fp-qpq9jsLc)

</div>