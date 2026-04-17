# readme_correction.md

**Rubric-to-file map.** Professor Ammar asked for this file explicitly —
every criterion in the project rubric is mapped below to its source of
evidence in this repository.

> Rubric codes follow the spreadsheet `project_rubric_assessment_google_sheets.xlsm`.
> File paths are repo-relative; line numbers are approximate and pinned
> to the version committed alongside this document.

---

## 1. Problem & Fit (15 %)

| Code | Criterion | Evidence |
|------|-----------|----------|
| PF1  | Specific problem / question | `BekaaSense_ProblemFormulation_v2.docx` §1 and `README.md` lines 1–24 |
| PF2A | User / decision / deployer | `BekaaSense_ProblemFormulation_v2.docx` §2; `README.md` "What the system does"; `docs/ARCHITECTURE.md` §"Deployment audience" |
| PF3A | Why ML, why not simpler | `BekaaSense_ProblemFormulation_v2.docx` §3; `docs/ARCHITECTURE.md` §"ML vs non-AI trade-off" |
| PF4  | Real-world impact | `BekaaSense_ProblemFormulation_v2.docx` §1–2 (Lebanon food-security framing); `docs/LIMITATIONS.md` §"Scope of claims" |
| PF5  | Type/track fit and success criteria | `BekaaSense_ProblemFormulation_v2.docx` header ("Option A"); `docs/EVALUATION.md` §"Success criteria" |

## 2. Technical Rigor & Responsible ML (30 %)

| Code | Criterion | Evidence |
|------|-----------|----------|
| TM1  | Task + data formulation | `data_ingestion/loaders.py` (`CANONICAL_COLS`), `data_ingestion/features.py` (`build_features`) |
| TM2A | Explicit non-AI baseline | `model_engine/baselines.py` — **three baselines** (`LinearTrendBaseline`, `SarimaBaseline`, `RuleBaseline`), all benchmarked in `model_engine/train.py` |
| TM3  | ML method choice + substance | `model_engine/ml_models.py` (RF, XGBoost, SMOTE classifier). Choice justified in `docs/ARCHITECTURE.md` §"Model selection" |
| TM4  | Preprocessing / features / **leakage** | `data_ingestion/cleaners.py` (impute + outlier flag), `data_ingestion/features.py` (shift-1 rollings), `tests/test_features.py` (`test_assert_no_leakage_fails_when_shuffled`, `test_forecast_features_do_not_include_target_current_value`) |
| TM5  | Splits, metrics, protocol | `data_ingestion/features.py::temporal_split`, `model_engine/evaluate.py`, `model_engine/train.py` (train 2015–2021, val 2022, test 2023→) |
| TM6  | Error analysis | `model_engine/evaluate.py::residuals_by_year`, `residuals_by_station`; outputs in `results/metrics/residuals_by_{year,station}.csv` |
| TM7  | Limitations + trade-offs | `docs/LIMITATIONS.md` + `README.md` §"Honest limitations" + dashboard limitations section |
| RM1  | Explainability | `model_engine/explainability.py` (SHAP TreeExplainer); API `/api/explain/`; dashboard SHAP chart |
| RM2  | Bias / fairness | Per-station residuals (`residuals_by_station.csv`); imputation flags in `data_ingestion/cleaners.py::add_imputation_flags`; station-scope disclaimer in `docs/LIMITATIONS.md` |
| RM3  | Privacy / data leakage | `.gitignore` excludes `data/raw/`; no PII in processed schema; feature-leakage guardrail test in `tests/test_features.py` |
| RM4  | Robustness / distribution shift | Bootstrapped 90 % prediction intervals (`ml_models.py::predict_with_interval`); coverage metric (`evaluate.py::interval_coverage`); intervals widen with horizon by construction (`inference.py::forecast_station`) |

## 3. Deployment & Engineering (20 %)

| Code | Criterion | Evidence |
|------|-----------|----------|
| EN1  | Dockerized API | `Dockerfile` (multi-stage, non-root, healthcheck); `docker-compose.yml` |
| EN2  | Separation of data / model / serving | 4 Django apps: `data_ingestion`, `model_engine`, `api`, `dashboard` — each with a single responsibility |
| EN3  | Reproducible env + run path | Pinned `requirements.txt`; `Makefile` targets (`install`, `data`, `train`, `test`, `up`); `random_state=42` throughout |
| EN4  | Functional UI / demo flow | `dashboard/` (Django templates + Chart.js dashboard with station selector, forecast chart, confidence band, SHAP chart, viability traffic light) |
| EN5  | Running deployed artefact | `docker compose up` starts the stack; `/health/` returns 200; `Dockerfile` HEALTHCHECK directive |

## 4. GitHub & Documentation (15 %)

| Code | Criterion | Evidence |
|------|-----------|----------|
| GD1  | Repo structure | See tree in `README.md` §"Repository layout" |
| GD2  | README: setup + run | `README.md` §"Quick start" (Docker + local + Makefile) |
| GD3  | Method / architecture docs | `docs/ARCHITECTURE.md` |
| GD4  | Results / logs / ablations | `results/metrics/leaderboard.csv`, `results/metrics/test_predictions.csv`, `results/metrics/shap_importance.csv`, residual CSVs, logger output in `bekaasense.settings::LOGGING` |
| GD5  | Data sources + limitations + deployment notes | `docs/DATA.md`, `docs/LIMITATIONS.md`, `README.md` |

## 5. Presentation (10 %) — graded at the poster session

| Code | Criterion | Prep file |
|------|-----------|-----------|
| PR1  | Problem framing clarity | `docs/LIMITATIONS.md` + problem formulation §1 |
| PR2  | Method / architecture explanation | `docs/ARCHITECTURE.md` |
| PR3  | Results / demo / visuals | Live dashboard at `/` + `results/metrics/leaderboard.csv` |
| PR4  | Q&A + ownership | Team preparation — not a repo artefact |

## 6. Creativity & Initiative (10 %)

| Code | Criterion | Evidence |
|------|-----------|----------|
| CI1  | Originality | No published ML work known for monthly desertification forecasting at Bekaa station level — `README.md` + problem formulation §1 |
| CI2  | Design trade-offs | `docs/ARCHITECTURE.md` §"Design decisions" + model choices documented in `ml_models.py` docstrings |
| CI3  | Beyond the minimum | **3** baselines (not 1); SHAP explainability; bootstrapped intervals with coverage; crop-viability traffic light; SMOTE + class-weighted classifier for class imbalance; Mann–Kendall honest disclosure |
| CI4  | Purposeful polish | Non-root Docker user; multi-stage image; healthcheck; CORS config; whitenoise static serving; env-driven settings; full test suite with leakage guardrail |

## 7. Bonus (+3 max)

| Code | Status | Notes |
|------|--------|-------|
| BX1  | Not attempted | Edge deployment — not applicable to a station-level climate-modelling system |
| BX2  | Attempted | RM beyond minimum: all 4 RM dimensions addressed (minimum is 3); interval coverage metric is a second robustness check beyond residual analysis |
| BX3  | Attempted | Three non-AI baselines (not one), SHAP surfaced in the UI (not just CSV), per-station residual stratification, class-imbalance handling (SMOTE + class_weight="balanced") |
