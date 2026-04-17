"""Feature engineering tests — the leakage guardrail lives here."""
import numpy as np
import pandas as pd
import pytest

from data_ingestion.features import (
    FORECAST_FEATURE_SET,
    assert_no_leakage,
    build_features,
    temporal_split,
)


def test_build_features_adds_expected_columns(synthetic_df):
    out = build_features(synthetic_df)
    expected = {"de_martonne", "aridity_zone", "month_sin", "month_cos",
                "dm_lag1", "dm_lag2", "dm_lag3", "precip_roll3",
                "dm_roll3", "spi3"}
    assert expected.issubset(out.columns), \
        f"missing: {expected - set(out.columns)}"


def test_temporal_split_is_strictly_ordered(synthetic_df):
    feat = build_features(synthetic_df)
    train, val, test = temporal_split(feat,
                                      train_end="2021-12-31",
                                      val_end="2022-12-31")
    if not train.empty and not val.empty:
        assert train["date"].max() < val["date"].min()
    if not val.empty and not test.empty:
        assert val["date"].max() < test["date"].min()


def test_assert_no_leakage_passes_for_valid_split(synthetic_df):
    feat = build_features(synthetic_df)
    train, _, test = temporal_split(feat,
                                    train_end="2021-12-31",
                                    val_end="2022-12-31")
    # Should not raise
    assert_no_leakage(train, test)


def test_assert_no_leakage_fails_when_shuffled(synthetic_df):
    feat = build_features(synthetic_df)
    feat["date"] = pd.to_datetime(
        dict(year=feat.year, month=feat.month, day=1)
    )
    # Deliberately build a bad split (random)
    bad_train = feat.sample(frac=0.7, random_state=1)
    bad_test = feat.drop(bad_train.index)
    with pytest.raises(AssertionError):
        assert_no_leakage(bad_train, bad_test)


def test_lag_feature_uses_only_past(synthetic_df):
    """For a given row, dm_lag1 must equal de_martonne of the PREVIOUS row
    for the same station."""
    feat = build_features(synthetic_df)
    for station, g in feat.groupby("station"):
        g = g.sort_values(["year", "month"]).reset_index(drop=True)
        # Compare rows 1..N
        for i in range(1, len(g)):
            if pd.isna(g.loc[i, "dm_lag1"]):
                continue
            assert abs(g.loc[i, "dm_lag1"] - g.loc[i - 1, "de_martonne"]) < 1e-9


def test_forecast_features_do_not_include_target_current_value():
    """The forecast-feature set must NOT contain de_martonne, precip_sum,
    or temp_avg (these are current-step values that would leak the target
    at inference time for future months)."""
    leaky = {"de_martonne", "precip_sum", "temp_avg", "temp_max", "temp_min"}
    assert not (set(FORECAST_FEATURE_SET) & leaky), \
        "FORECAST_FEATURE_SET contains leaky current-step variables."
