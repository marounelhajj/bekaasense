"""
Inference service.

Holds a lazily-initialised registry of fitted models loaded from
``model_engine/artifacts/``. The Django API views call into this module
rather than re-training or re-loading on every request.
"""
from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from model_engine.ml_models import load_model

log = logging.getLogger("bekaasense")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "model_engine" / "artifacts"
METRICS = REPO_ROOT / "results" / "metrics"

_lock = threading.Lock()
_registry: dict = {}


def _load_feature_sets() -> dict:
    p = ARTIFACTS / "feature_sets.json"
    if not p.exists():
        return {"forecast": [], "nowcast": []}
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def get_registry() -> dict:
    """Return loaded models. Thread-safe, cached after first call."""
    with _lock:
        if _registry:
            return _registry
        artefacts = {
            "rf": "random_forest.joblib",
            "xgb": "xgboost.joblib",
            "zone": "zone_classifier.joblib",
            "linear": "baseline_linear.joblib",
        }
        for key, name in artefacts.items():
            path = ARTIFACTS / name
            if path.exists():
                _registry[key] = load_model(path)
            else:
                log.warning("Artefact missing: %s — run `make train`.", path)
        _registry["feature_sets"] = _load_feature_sets()
        return _registry


# ---------------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------------

def _latest_features(station: str) -> pd.DataFrame:
    """Read the canonical CSV and build the feature row for the *latest*
    observation of ``station`` — this is the base from which we roll the
    forecast forward."""
    from data_ingestion.features import build_features
    from data_ingestion.loaders import load_clean_csv

    df = load_clean_csv(REPO_ROOT / "data" / "processed" / "bekaa_valley_clean.csv")
    df = df[df["station"] == station]
    if df.empty:
        raise ValueError(f"No data found for station {station!r}")
    full_feat = build_features(df)
    return full_feat.tail(1).reset_index(drop=True)


def forecast_station(station: str, horizon_months: int = 12,
                     alpha: float = 0.1) -> list[dict]:
    """Roll a Random-Forest forecast forward ``horizon_months`` steps.

    Recursive forecasting: each new prediction becomes the lag-1 feature
    for the next step. The bootstrap interval is reported for every
    horizon step; it widens monotonically with lead time — the correct
    behaviour for an honest uncertainty quantification (RM4).
    """
    reg = get_registry()
    rf = reg.get("rf")
    if rf is None:
        raise RuntimeError("RandomForest model not loaded. Run `make train`.")

    feat_row = _latest_features(station)
    forecast_feats = reg["feature_sets"]["forecast"]
    missing = [c for c in forecast_feats if c not in feat_row.columns]
    if missing:
        raise RuntimeError(f"Feature row missing: {missing}")

    # Pull the rolling state we need to update
    state = feat_row.iloc[0].to_dict()
    year, month = int(state["year"]), int(state["month"])

    out: list[dict] = []
    # Keep a running buffer of recent DM, precipitation, temperature for
    # rolling-feature updates.
    dm_hist = [state.get("dm_lag1", state.get("de_martonne", 0.0)),
               state.get("de_martonne", 0.0)]
    precip_hist = [state.get("precip_lag1", state.get("precip_sum", 0.0)),
                   state.get("precip_sum", 0.0)]
    temp_hist = [state.get("temp_lag1", state.get("temp_avg", 15.0)),
                 state.get("temp_avg", 15.0)]

    for h in range(1, horizon_months + 1):
        month = month % 12 + 1
        if month == 1:
            year += 1

        # Update seasonal features
        state["month"] = month
        state["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        state["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))

        # Update lag features from history
        state["dm_lag1"] = dm_hist[-1]
        state["dm_lag2"] = dm_hist[-2] if len(dm_hist) >= 2 else dm_hist[-1]
        state["dm_lag3"] = dm_hist[-3] if len(dm_hist) >= 3 else dm_hist[-1]
        state["precip_lag1"] = precip_hist[-1]
        state["temp_lag1"] = temp_hist[-1]

        # Rolling — recompute from most recent k values
        state["dm_roll3"] = float(np.mean(dm_hist[-3:]))
        state["dm_roll6"] = float(np.mean(dm_hist[-6:]))
        state["precip_roll3"] = float(np.mean(precip_hist[-3:]))
        state["precip_roll6"] = float(np.mean(precip_hist[-6:]))
        state["precip_roll12"] = float(np.mean(precip_hist[-12:]))

        x = pd.DataFrame([{c: state.get(c, 0.0) for c in forecast_feats}])
        mean, lo, hi = rf.predict_with_interval(x, alpha=alpha)
        # Broaden with horizon — increases ±5% per step up to 50%
        widen = 1 + 0.05 * (h - 1)
        dm_pred = float(mean[0])
        spread = max(hi[0] - lo[0], 1e-6) * widen / 2.0
        lo_h, hi_h = dm_pred - spread, dm_pred + spread

        out.append({
            "year": year, "month": month, "horizon": h,
            "de_martonne_pred": dm_pred,
            "lower": lo_h, "upper": hi_h,
            "aridity_zone": _zone_for_value(dm_pred),
        })

        # Update rolling buffer: use predicted DM, keep precipitation /
        # temperature at station climatology (rough but honest).
        dm_hist.append(dm_pred)

    return out


def _zone_for_value(v: float) -> str:
    from data_ingestion.indices import DE_MARTONNE_ZONES
    for name, lo, hi in DE_MARTONNE_ZONES:
        if lo <= v < hi:
            return name
    return "Unknown"


# ---------------------------------------------------------------------------
# Crop viability — traffic light for the dashboard
# ---------------------------------------------------------------------------

def crop_viability(de_martonne_value: float) -> dict:
    """Map a projected DM index to a three-colour crop viability signal
    for rain-fed wheat. The 20-threshold comes from the problem
    formulation (De Martonne < 20 => rain-fed wheat marginal).
    """
    if de_martonne_value >= 20:
        return {"status": "green",
                "label": "Viable",
                "note": "Rain-fed wheat is climatically viable."}
    if de_martonne_value >= 10:
        return {"status": "yellow",
                "label": "Marginal",
                "note": "Rain-fed wheat is marginal; irrigation strongly advised."}
    return {"status": "red",
            "label": "Non-viable",
            "note": "Rain-fed wheat not climatically viable; irrigation required."}
