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
5. [Quick Start](#5-quick-start)
6. [API Reference](#6-api-reference)
7. [System Capabilities](#7-system-capabilities)

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

- **Temporal scope:** 11 years limits long-horizon reliability. Intervals widen with horizon by design.
- **Spatial scope:** 4 stations; no claims for unmonitored locations.
- **Index simplicity:** De Martonne uses only precipitation and temperature — no wind, solar
  radiation, or actual evapotranspiration. SPEI would be a more complete indicator.
- **Distribution shift:** The climate of 2015–2025 may not represent 2030+. The system mitigates
  this through horizon-widening intervals and year-stratified residual monitoring.
- **Trend significance:** Mann–Kendall tests return p > 0.05 on all four stations. Directional
  trends are consistent with regional climatology but not yet statistically significant at 11 years.

### Future Work

| Priority | Direction |
|---|---|
| 🔴 High | Extend record with ERA5 reanalysis back to 1980+ |
| 🔴 High | Add SPEI as a second target — more complete than De Martonne |
| 🟡 Medium | Satellite covariates (NDVI, soil moisture from Sentinel/MODIS) |
| 🟡 Medium | Temporal Fusion Transformer for improved long-horizon forecasting |
| 🟢 Low | Real-time data pipeline from LARI when API becomes available |

---

## 5. Quick Start

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

## 6. API Reference

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

## 7. System Capabilities

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