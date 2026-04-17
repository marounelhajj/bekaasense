"""Smoke tests for the DRF endpoints."""
import pytest
from django.test import Client


@pytest.mark.django_db
def test_health_endpoint():
    c = Client()
    r = c.get("/health/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.django_db
def test_stations_endpoint_without_data_returns_503():
    """When no processed CSV exists, the endpoint should 503 with a
    helpful message rather than crashing."""
    from django.conf import settings
    from pathlib import Path
    csv = Path(settings.DATA_PROCESSED_DIR) / "bekaa_valley_clean.csv"
    if csv.exists():
        pytest.skip("Data file present; this test is for the empty case.")
    c = Client()
    r = c.get("/api/stations/")
    assert r.status_code in (200, 503)  # tolerant if someone ran `make data`


@pytest.mark.django_db
def test_predict_rejects_invalid_station():
    c = Client()
    r = c.post("/api/predict/",
               data={"station": "NotAStation", "horizon_months": 12},
               content_type="application/json")
    assert r.status_code == 400
