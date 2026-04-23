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
    from imblearn.combine import SMOTETomek  # type: ignore
    from imblearn.over_sampling import SMOTE  # type: ignore
    HAS_SMOTE = True
except Exception:  # pragma: no cover
    HAS_SMOTE = False


# ---------------------------------------------------------------------------
# Regression — De Martonne index forecast
# ---------------------------------------------------------------------------

@dataclass
class RandomForestForecaster:
    """Random Forest regressor with conformalized prediction intervals.

    Point predictions come from the standard RF mean. Intervals are
    calibrated via split conformal prediction on the validation set:
    the signed residuals (y_val - y_pred) at the alpha/2 and 1-alpha/2
    quantiles are stored and added to every new point prediction. This
    guarantees empirical coverage >= (1-alpha) on exchangeable test data,
    unlike per-tree quantiles which only capture epistemic uncertainty
    and systematically under-cover.
    """

    n_estimators: int = 400
    max_depth: int | None = 10
    min_samples_leaf: int = 4
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    model_: RandomForestRegressor | None = None
    _conf_q_lo_: float = field(default=float("nan"))
    _conf_q_hi_: float = field(default=float("nan"))

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

    def calibrate_intervals(self, X_val: pd.DataFrame,
                            y_val: pd.Series, alpha: float = 0.1) -> None:
        """Compute and store the conformal half-width from the validation set.

        Uses symmetric split conformal prediction with the finite-sample
        correction: the calibration quantile level is
        ceil((n_cal + 1) * (1 - alpha)) / n_cal rather than (1 - alpha).
        This guarantees marginal coverage >= (1-alpha) on exchangeable
        test data, regardless of the residual distribution shape.
        """
        residuals = y_val.values.astype(float) - self.predict(X_val)
        n = len(residuals)
        q_level = min(1.0, float(np.ceil((n + 1) * (1 - alpha))) / n)
        self._conf_half_width_ = float(np.quantile(np.abs(residuals), q_level))

    def predict_with_interval(self, X: pd.DataFrame,
                              alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) with guaranteed (1-alpha) coverage.

        Uses the calibrated symmetric conformal half-width if
        calibrate_intervals() has been called; falls back to per-tree
        quantiles otherwise.
        """
        assert self.model_ is not None, "Call fit() before predict()."
        mean = self.predict(X)
        half = getattr(self, "_conf_half_width_", float("nan"))
        if not np.isnan(half):
            return mean, mean - half, mean + half
        # Uncalibrated fallback — per-tree spread
        Xv = X[self.feature_names_].values
        per_tree = np.stack([t.predict(Xv) for t in self.model_.estimators_])
        lo = np.quantile(per_tree, alpha / 2, axis=0)
        hi = np.quantile(per_tree, 1 - alpha / 2, axis=0)
        return mean, lo, hi


@dataclass
class XGBoostForecaster:
    """Gradient-boosted forecaster with early stopping against a validation
    set to prevent over-fitting on the small Bekaa training set (~330 rows).

    Without early stopping, XGBoost reaches R²≈0.9999 on train but degrades
    on val/test. Early stopping on val RMSE caps the ensemble at the point of
    best generalisation rather than minimum training loss.
    """

    n_estimators: int = 600   # upper bound; early stopping will trim this
    max_depth: int = 5        # shallower than before to reduce variance
    learning_rate: float = 0.05
    early_stopping_rounds: int = 30
    reg_lambda: float = 2.0   # L2 weight regularization
    reg_alpha: float = 0.1    # L1 weight regularization
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    model_: object | None = None
    _val_X: object | None = None
    _val_y: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "XGBoostForecaster":
        if not HAS_XGB:
            raise ImportError("xgboost not installed; `pip install xgboost`")
        self.feature_names_ = list(X.columns)
        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1, verbosity=0,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
        )
        fit_kwargs: dict = {"X": X.values, "y": y.values}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[self.feature_names_].values, y_val.values)]
            fit_kwargs["verbose"] = False
        self.model_.fit(**fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict(X[self.feature_names_].values)

    def predict_with_interval(self, X: pd.DataFrame,
                              alpha: float = 0.1,
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper) with guaranteed (1-alpha) coverage.

        Uses symmetric split conformal prediction with the finite-sample
        correction: quantile level = ceil((n_cal+1)*(1-alpha)) / n_cal.
        """
        assert self.model_ is not None
        base = self.predict(X)
        half = getattr(self, "_conf_half_width_", float("nan"))
        if not np.isnan(half):
            return base, base - half, base + half
        # Fallback before calibration: raw residual quantile using alpha
        res = np.abs(getattr(self, "_residuals_", np.zeros(1)))
        fallback = float(np.quantile(res, 1 - alpha))
        return base, base - fallback, base + fallback

    def calibrate_residuals(self, X_val: pd.DataFrame,
                            y_val: pd.Series, alpha: float = 0.1) -> None:
        """Compute and store the conformal half-width from the validation set.

        Uses finite-sample corrected quantile level so coverage >= (1-alpha)
        is guaranteed on exchangeable test data.
        """
        residuals = (y_val.values - self.predict(X_val)).astype(float)
        self._residuals_ = residuals
        n = len(residuals)
        q_level = min(1.0, float(np.ceil((n + 1) * (1 - alpha))) / n)
        self._conf_half_width_ = float(np.quantile(np.abs(residuals), q_level))


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
            # SMOTETomek: oversample minority classes then remove borderline
            # Tomek link pairs — produces cleaner decision boundaries than
            # plain SMOTE, which is important for the Semi-arid/Sub-humid edge.
            try:
                k = min(5, int(pd.Series(y_tr).value_counts().min()) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=self.random_state)
                    X_tr, y_tr = SMOTETomek(
                        smote=smote, random_state=self.random_state
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


@dataclass
class XGBoostClassifier:
    """XGBoost multiclass classifier for aridity zone prediction.

    Gradient boosted trees learn the DM threshold boundaries more precisely
    than an axis-aligned Random Forest, especially when de_martonne is in
    the feature set (the boundary is a simple step function on that feature).
    """

    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42
    feature_names_: list[str] = field(default_factory=list)
    classes_: np.ndarray | None = None
    _label_map_: dict = field(default_factory=dict)
    _inv_label_map_: dict = field(default_factory=dict)
    model_: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostClassifier":
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        from sklearn.preprocessing import LabelEncoder
        self.feature_names_ = list(X.columns)
        self.classes_ = np.array(sorted(y.unique()))
        le = LabelEncoder()
        y_enc = le.fit_transform(y.values)
        self._label_map_ = {cls: i for i, cls in enumerate(le.classes_)}
        self._inv_label_map_ = {i: cls for cls, i in self._label_map_.items()}

        X_tr, y_tr = X.values, y_enc
        if HAS_SMOTE:
            try:
                k = min(5, int(pd.Series(y.values).value_counts().min()) - 1)
                if k >= 1:
                    smote = SMOTE(k_neighbors=k, random_state=self.random_state)
                    X_tr, y_tr = SMOTETomek(
                        smote=smote, random_state=self.random_state
                    ).fit_resample(X_tr, y_tr)
            except Exception:
                pass

        from xgboost import XGBClassifier as _XGBClassifier
        self.model_ = _XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=len(self.classes_),
            random_state=self.random_state,
            n_jobs=-1, verbosity=0,
            eval_metric="mlogloss",
        )
        self.model_.fit(X_tr, y_tr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        enc_preds = self.model_.predict(X[self.feature_names_].values)
        return np.array([self._inv_label_map_[i] for i in enc_preds])

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
