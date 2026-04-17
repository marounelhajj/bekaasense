"""Tests for :mod:`data_ingestion.indices`."""
import numpy as np
import pandas as pd

from data_ingestion.indices import (
    classify_aridity,
    de_martonne_monthly,
    spi,
)


def test_de_martonne_monotonic_in_precip():
    """Higher precipitation at fixed temperature => higher index."""
    t = pd.Series([15.0, 15.0, 15.0])
    p = pd.Series([10.0, 50.0, 100.0])
    idx = de_martonne_monthly(p, t)
    assert (idx.diff().dropna() > 0).all()


def test_de_martonne_monotonic_in_temp_inverse():
    """Higher temperature at fixed precipitation => lower index."""
    p = pd.Series([50.0, 50.0, 50.0])
    t = pd.Series([5.0, 15.0, 25.0])
    idx = de_martonne_monthly(p, t)
    assert (idx.diff().dropna() < 0).all()


def test_de_martonne_handles_very_cold_temp():
    """Temperatures near -10°C must not blow up the denominator."""
    p = pd.Series([30.0])
    t = pd.Series([-12.0])  # clipped to -9.5
    idx = de_martonne_monthly(p, t)
    assert np.isfinite(idx.iloc[0])
    assert idx.iloc[0] > 0


def test_classify_aridity_thresholds():
    vals = pd.Series([2.0, 7.0, 15.0, 25.0, 40.0])
    zones = classify_aridity(vals).tolist()
    assert zones == ["Hyper-arid", "Arid", "Semi-arid", "Sub-humid", "Humid"]


def test_spi_centered_near_zero():
    """Long stationary series => SPI should be near zero in expectation."""
    rng = np.random.default_rng(0)
    p = pd.Series(rng.gamma(2.0, 20.0, size=200))
    s = spi(p, window=3).dropna()
    assert abs(s.mean()) < 0.4  # loose; finite-sample bias


def test_spi_negative_under_drought():
    """Abrupt drop in precipitation => negative SPI."""
    p = pd.Series([40.0] * 60 + [2.0] * 20)
    s = spi(p, window=3)
    assert s.iloc[-1] < -0.5
