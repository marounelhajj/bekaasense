"""Non-AI baseline sanity tests."""
import numpy as np
import pandas as pd

from data_ingestion.features import build_features
from model_engine.baselines import (
    LinearTrendBaseline,
    RuleBaseline,
    SarimaBaseline,
)


def test_linear_trend_fits_and_predicts(synthetic_df):
    feat = build_features(synthetic_df)
    model = LinearTrendBaseline().fit(feat)
    pred = model.predict(feat)
    assert len(pred) == len(feat)
    assert np.all(np.isfinite(pred))


def test_sarima_fits_or_falls_back(synthetic_df):
    # With < 24 months per station the baseline falls back to the mean.
    feat = build_features(synthetic_df)
    model = SarimaBaseline().fit(feat)
    pred = model.predict(feat)
    assert len(pred) == len(feat)
    assert np.all(np.isfinite(pred))


def test_rule_baseline_returns_valid_labels(synthetic_df):
    feat = build_features(synthetic_df).dropna(subset=["precip_roll3"])
    model = RuleBaseline().fit(feat)
    preds = model.predict(feat)
    assert set(preds).issubset({"drier-than-normal", "normal-or-wetter"})
