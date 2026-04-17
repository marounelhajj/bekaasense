"""
ML models for BekaaSense.

Two tasks:

* **Regression** — forecast the monthly De Martonne index. Primary models
  are :class:`RandomForestForecaster` and :class:`XGBoostForecaster`.
  Both expose ``predict_with_interval`` for bootstrapped confidence bands
  (responsible-ML requirement RM4: robustness under uncertainty).

* **Classification** — predict the aridity zone of a future month. Uses a
  class-weighted :class:`RandomForestClassifier` combined optionally with
  SMOTE oversampling, motivated by the EDA finding that Hyper-arid months
  dominate the dataset.

All models follow the scikit-learn ``fit`` / ``predict`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_SMOTE = True
except Exception:  # pragma: no cover
    HAS_SMOTE = False


# ---------------------------------------------------------------------------
# Regression — De Martonne index forecast
# ---------------------------------------------------------------------------

@dataclass
class RandomForestForecaster:
    """Random Forest regressor with bootstrapped prediction intervals.

    Intervals are computed from the per-tree predictions, which the
    ensemble structure exposes directly — no external bootstrap needed.
    """

    n_estimators: int = 400
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    model_: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self.feature_names_ = list(X.columns)
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "Call fit() before predict()."
        return self.model_.predict(X[self.feature_names_].values)

    def predict_with_interval(self, X: pd.DataFrame,
                              alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) with ``(1-alpha)`` coverage.

        Uses the empirical distribution of per-tree predictions. For a
        forest of 400 trees with alpha=0.1, bounds are the 5th / 95th
        per-row percentiles.
        """
        assert self.model_ is not None, "Call fit() before predict()."
        Xv = X[self.feature_names_].values
        per_tree = np.stack([t.predict(Xv) for t in self.model_.estimators_])
        mean = per_tree.mean(axis=0)
        lo = np.quantile(per_tree, alpha / 2, axis=0)
        hi = np.quantile(per_tree, 1 - alpha / 2, axis=0)
        return mean, lo, hi


@dataclass
class XGBoostForecaster:
    """Gradient-boosted forecaster. Often marginally wins over RF on this
    dataset; kept as a second model for ensembling / comparison."""

    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    model_: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        if not HAS_XGB:
            raise ImportError("xgboost not installed; `pip install xgboost`")
        self.feature_names_ = list(X.columns)
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1, verbosity=0,
        )
        self.model_.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict(X[self.feature_names_].values)

    def predict_with_interval(self, X: pd.DataFrame,
                              alpha: float = 0.1,
                              n_bootstrap: int = 80,
                              random_state: int = 42
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Residual bootstrap intervals.

        XGBoost does not expose per-tree predictions like RF, so we use a
        simpler residual-bootstrap approximation: resample training
        residuals, add them to the point forecast, and take percentiles.
        """
        assert self.model_ is not None
        rng = np.random.default_rng(random_state)
        base = self.predict(X)
        residuals = getattr(self, "_residuals_", np.zeros(1))
        samples = rng.choice(residuals, size=(n_bootstrap, len(X)), replace=True)
        boots = base[None, :] + samples
        lo = np.quantile(boots, alpha / 2, axis=0)
        hi = np.quantile(boots, 1 - alpha / 2, axis=0)
        return base, lo, hi

    def calibrate_residuals(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Store validation residuals for the bootstrap interval."""
        self._residuals_ = (y_val.values - self.predict(X_val)).astype(float)


# ---------------------------------------------------------------------------
# Classification — aridity zone
# ---------------------------------------------------------------------------

@dataclass
class AridityZoneClassifier:
    """Random Forest classifier with optional SMOTE for class imbalance.

    EDA shows Hyper-arid dominates ~80% of months across stations. Without
    re-balancing, F1 on minority (Humid / Sub-humid) classes collapses.
    """

    use_smote: bool = True
    class_weight: str | dict = "balanced"
    n_estimators: int = 400
    max_depth: int | None = 14
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    classes_: np.ndarray | None = None
    model_: RandomForestClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AridityZoneClassifier":
        self.feature_names_ = list(X.columns)

        X_tr, y_tr = X.values, y.values
        if self.use_smote and HAS_SMOTE:
            # SMOTE only for classes with > 1 sample
            try:
                k = min(5, int(pd.Series(y_tr).value_counts().min()) - 1)
                if k >= 1:
                    X_tr, y_tr = SMOTE(
                        k_neighbors=k, random_state=self.random_state
                    ).fit_resample(X_tr, y_tr)
            except Exception:
                pass  # fallback: class weights alone

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight=self.class_weight, n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(X_tr, y_tr)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X[self.feature_names_].values)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str):
    return joblib.load(path)
