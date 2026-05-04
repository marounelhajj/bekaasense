# readme_correction.md

**Implementation map.** Every feature and design decision in this project is mapped to its source of evidence in this repository.

> File paths are repo-relative; line numbers are approximate and pinned to the version committed alongside this document.

---

## 1. Problem & Fit

| Criterion | Evidence |
|-----------|----------|
| Specific problem / question | `BekaaSense_ProblemFormulation_v2.docx` §1 and `README.md` lines 1–24 |
| User / decision / deployer | `BekaaSense_ProblemFormulation_v2.docx` §2; `README.md` "What the system does"; `docs/ARCHITECTURE.md` §"Deployment audience" |
| Why ML, why not simpler | `BekaaSense_ProblemFormulation_v2.docx` §3; `docs/ARCHITECTURE.md` §"ML vs non-AI trade-off" |
| Real-world impact | `BekaaSense_ProblemFormulation_v2.docx` §1–2 (Lebanon food-security framing); `docs/LIMITATIONS.md` §"Scope of claims" |
| Type/track fit and success criteria | `BekaaSense_ProblemFormulation_v2.docx` header ("Option A"); `docs/EVALUATION.md` §"Success criteria" |

## 2. Technical Rigor & Responsible ML

| Criterion | Evidence |
|-----------|----------|
| Task + data formulation | `data_ingestion/loaders.py` (`CANONICAL_COLS`), `data_ingestion/features.py` (`build_features`) |
| Explicit non-AI baseline | `model_engine/baselines.py` — **three baselines** (`LinearTrendBaseline`, `SarimaBaseline`, `RuleBaseline`), all benchmarked in `model_engine/train.py` |
| ML method choice + substance | `model_engine/ml_models.py` (RF, XGBoost regressors; XGBoostClassifier with SMOTE-Tomek). Choice justified in `docs/ARCHITECTURE.md` §"Model selection" |
| Preprocessing / features / **leakage** | `data_ingestion/cleaners.py` (impute + outlier flag), `data_ingestion/features.py` (shift-1 rollings), `tests/test_features.py` (`test_assert_no_leakage_fails_when_shuffled`, `test_forecast_features_do_not_include_target_current_value`) |
| Splits, metrics, protocol | `data_ingestion/features.py::temporal_split`, `model_engine/evaluate.py`, `model_engine/train.py` (train 2015–2021, val 2022, test 2023→) |
| Error analysis | `model_engine/evaluate.py::residuals_by_year`, `residuals_by_station`; outputs in `results/metrics/residuals_by_{year,station}.csv`; surfaced in dashboard via `/api/residuals/` as grouped bar charts (bias + MAE by station, MAE by year) |
| Limitations + trade-offs | `docs/LIMITATIONS.md` + `README.md` §"Honest limitations" + dashboard limitations section |
| Explainability | `model_engine/explainability.py` (SHAP TreeExplainer); API `/api/explain/`; dashboard SHAP chart |
| Bias / fairness | Per-station residuals (`residuals_by_station.csv`); imputation flags in `data_ingestion/cleaners.py::add_imputation_flags`; station-scope disclaimer in `docs/LIMITATIONS.md` |
| Privacy / data leakage | `.gitignore` excludes `data/raw/`; no PII in processed schema; feature-leakage guardrail test in `tests/test_features.py` |
| Robustness / distribution shift | Split conformal prediction intervals with finite-sample correction (`ml_models.py::calibrate_intervals`, `calibrate_residuals`); coverage metric (`evaluate.py::interval_coverage`); actual test coverage RF 93.8%, XGB 94.4% vs 90% target; intervals widen with horizon by construction (`inference.py::forecast_station`); `model_health.json` flags any model failing quality thresholds post-training |

## 3. Deployment & Engineering

| Criterion | Evidence |
|-----------|----------|
| Dockerized API | `Dockerfile` (multi-stage, non-root, healthcheck); `docker-compose.yml` |
| Separation of data / model / serving | 4 Django apps: `data_ingestion`, `model_engine`, `api`, `dashboard` — each with a single responsibility |
| Reproducible env + run path | Pinned `requirements.txt`; `Makefile` targets (`install`, `data`, `train`, `test`, `up`); `random_state=42` throughout |
| Functional UI / demo flow | `dashboard/` (Django templates + Chart.js dashboard with station selector, forecast chart, confidence band, SHAP chart, viability traffic light) |
| Running deployed artefact | `docker compose up` starts the stack; `/health/` returns 200; `Dockerfile` HEALTHCHECK directive; live on Azure at `bekaasense-fqfdedgmdjcvh4g4.francecentral-01.azurewebsites.net` |
| CI/CD pipeline | `.github/workflows/ci.yml` — lint + synthetic data + train + 17 pytest tests on every push; `.github/workflows/main_bekaasense.yml` — auto-deploy to Azure App Service after CI passes |

## 4. Documentation

| Criterion | Evidence |
|-----------|----------|
| Repo structure | See tree in `README.md` §"Repository layout" |
| README: setup + run | `README.md` §"Quick start" (Docker + local + Makefile) |
| Method / architecture docs | `docs/ARCHITECTURE.md` |
| Results / logs / ablations | `results/metrics/leaderboard.csv`, `results/metrics/test_predictions.csv`, `results/metrics/shap_importance.csv`, residual CSVs, logger output in `bekaasense.settings::LOGGING` |
| Data sources + limitations + deployment notes | `docs/DATA.md`, `docs/LIMITATIONS.md`, `README.md` |

## 5. Creativity & Initiative

| Criterion | Evidence |
|-----------|----------|
| Originality | No published ML work known for monthly desertification forecasting at Bekaa station level — `README.md` + problem formulation §1 |
| Design trade-offs | `docs/ARCHITECTURE.md` §"Design decisions" + model choices documented in `ml_models.py` docstrings |
| Beyond the minimum | **3** baselines (not 1); SHAP explainability; conformal prediction intervals with coverage guarantee; crop-viability traffic light; SMOTE-Tomek + class-weighted XGBoostClassifier (RF and XGB both trained, best kept); Mann–Kendall honest disclosure; decision guide + scientific conclusions on dashboard; model health check system |
| Purposeful polish | Non-root Docker user; multi-stage image; healthcheck; CORS config; whitenoise static serving; env-driven settings; full test suite with leakage guardrail |

## 6. Extra Features & Scope

| Feature | Notes |
|---------|-------|
| All 4 Responsible-ML dimensions | Explainability (SHAP), Fairness/bias (per-station residuals), Privacy (leakage guardrail), Robustness (conformal intervals + model health check) |
| Three non-AI baselines | LinearTrend, SARIMA, Rule-based — not just one |
| SHAP in the UI | Interactive bar chart in dashboard, not just a CSV |
| Error analysis in dashboard | `/api/residuals/` + two charts: spatial bias by station and temporal drift by year |
| CI/CD pipeline | GitHub Actions: 17 automated tests on every push before Azure deploys |
| Edge deployment | Not applicable to a station-level climate-modelling system — not implemented |
| Real-time data ingestion | LARI has no public API — not implemented |
| Spatial interpolation | Only 4 stations, no grid coverage — not implemented |