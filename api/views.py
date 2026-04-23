"""
DRF API views.

Endpoints (all under ``/api/``):

* ``GET  /stations/``               — metadata for every station
* ``POST /predict/``                — future De Martonne forecast + interval
* ``POST /classify/``               — aridity-zone classification
* ``GET  /trend/?station=<name>``   — historical series for the dashboard
* ``POST /explain/``                — SHAP explanation for the latest row
* ``GET  /leaderboard/``            — comparison table of all models
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.serializers import (
    ClassifyRequestSerializer,
    ExplainRequestSerializer,
    ForecastRequestSerializer,
    VALID_STATIONS,
)

log = logging.getLogger("bekaasense")

METRICS = Path(settings.RESULTS_DIR) / "metrics"
DATA_CSV = Path(settings.DATA_PROCESSED_DIR) / "bekaa_valley_clean.csv"


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------

@api_view(["GET"])
def stations(request):
    """List monitored stations with latest observation + current aridity zone."""
    from data_ingestion.features import build_features
    from data_ingestion.loaders import load_clean_csv
    from model_engine.inference import crop_viability

    if not DATA_CSV.exists():
        return Response(
            {"detail": "Processed dataset not found. "
                       "Run `make data` or `python scripts/generate_synthetic.py`."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    df = load_clean_csv(DATA_CSV)
    feat = build_features(df)
    latest = (feat.sort_values("date")
                  .groupby("station")
                  .tail(1)
                  .reset_index(drop=True))

    payload = []
    for row in latest.itertuples(index=False):
        payload.append({
            "station": row.station,
            "latest_year": int(row.year),
            "latest_month": int(row.month),
            "latest_precip_mm": float(row.precip_sum),
            "latest_temp_c": float(row.temp_avg),
            "latest_de_martonne": float(row.de_martonne),
            "current_zone": str(row.aridity_zone),
            "crop_viability": crop_viability(float(row.de_martonne)),
        })
    return Response({"stations": payload})


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@api_view(["POST"])
def predict(request):
    s = ForecastRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    params = s.validated_data

    try:
        from model_engine.inference import forecast_station
        forecast = forecast_station(
            params["station"],
            horizon_months=params["horizon_months"],
            alpha=params["alpha"],
        )
    except Exception as exc:
        log.exception("Forecast failed")
        return Response({"detail": str(exc)},
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({
        "station": params["station"],
        "horizon_months": params["horizon_months"],
        "nominal_coverage": 1 - params["alpha"],
        "forecast": forecast,
    })


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@api_view(["POST"])
def classify(request):
    """Classify a single (station, year, month) into an aridity zone.

    If the target point lies within observed history, the classifier
    answers directly from the feature row; otherwise we forecast forward
    and report the projected zone.
    """
    s = ClassifyRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    p = s.validated_data

    from data_ingestion.features import build_features
    from data_ingestion.loaders import load_clean_csv

    df = load_clean_csv(DATA_CSV)
    feat = build_features(df)
    mask = ((feat["station"] == p["station"]) &
            (feat["year"] == p["year"]) &
            (feat["month"] == p["month"]))
    if mask.any():
        row = feat[mask].iloc[0]
        return Response({
            "station": p["station"], "year": p["year"], "month": p["month"],
            "source": "observed",
            "de_martonne": float(row["de_martonne"]),
            "aridity_zone": str(row["aridity_zone"]),
        })

    # Out of range -> forward simulate
    from model_engine.inference import forecast_station
    latest = feat[feat["station"] == p["station"]].sort_values("date").tail(1).iloc[0]
    horizon = (p["year"] - int(latest["year"])) * 12 + (p["month"] - int(latest["month"]))
    if horizon <= 0:
        return Response({"detail": "Requested date is earlier than the latest "
                                  "observation but not present in the record. "
                                  "This likely means the data gap needs "
                                  "backfilling."},
                       status=status.HTTP_404_NOT_FOUND)
    fc = forecast_station(p["station"], horizon_months=horizon)
    if not fc:
        return Response({"detail": "Empty forecast."},
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    target = fc[-1]
    return Response({
        "station": p["station"], "year": p["year"], "month": p["month"],
        "source": "forecast",
        "de_martonne": target["de_martonne_pred"],
        "aridity_zone": target["aridity_zone"],
        "lower": target["lower"], "upper": target["upper"],
    })


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

@api_view(["GET"])
def trend(request):
    """Historical monthly De Martonne series for a given station."""
    station = request.query_params.get("station")
    if station not in VALID_STATIONS:
        return Response({"detail": f"station must be one of {VALID_STATIONS}"},
                       status=status.HTTP_400_BAD_REQUEST)

    from data_ingestion.features import build_features
    from data_ingestion.loaders import load_clean_csv
    df = load_clean_csv(DATA_CSV)
    feat = build_features(df)
    g = feat[feat["station"] == station].sort_values("date")
    history = [
        {"year": int(r.year), "month": int(r.month),
         "de_martonne": float(r.de_martonne),
         "precip_sum": float(r.precip_sum),
         "temp_avg": float(r.temp_avg),
         "aridity_zone": str(r.aridity_zone)}
        for r in g.itertuples(index=False)
    ]

    # Append the latest 12-month forecast so the UI can draw it continuously
    forecast: list = []
    try:
        from model_engine.inference import forecast_station
        forecast = forecast_station(station, horizon_months=24, alpha=0.1)
    except Exception as e:
        log.warning("Could not append forecast: %s", e)

    return Response({
        "station": station,
        "history": history,
        "forecast": forecast,
    })


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

@api_view(["POST"])
def explain(request):
    s = ExplainRequestSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    p = s.validated_data

    try:
        from data_ingestion.features import build_features
        from data_ingestion.loaders import load_clean_csv
        from model_engine.explainability import explain_prediction
        from model_engine.inference import get_registry

        reg = get_registry()
        rf = reg.get("rf")
        if rf is None:
            return Response({"detail": "Model not trained."},
                           status=status.HTTP_503_SERVICE_UNAVAILABLE)

        df = load_clean_csv(DATA_CSV)
        feat = build_features(df)
        # Use the latest row that has all forecast features populated
        forecast_feats = reg["feature_sets"]["forecast"]
        rows = feat[feat["station"] == p["station"]].dropna(subset=forecast_feats)
        if rows.empty:
            return Response({"detail": "No feature row with complete lag history."},
                           status=status.HTTP_503_SERVICE_UNAVAILABLE)
        row = rows.sort_values("date").tail(1)
        x_row = row[forecast_feats].reset_index(drop=True)
        result = explain_prediction(rf, x_row, top_k=p["top_k"])
        result["station"] = p["station"]
        return Response(result)
    except Exception as exc:
        log.exception("SHAP explain failed")
        return Response({"detail": str(exc)},
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@api_view(["GET"])
def leaderboard(request):
    p = METRICS / "leaderboard.json"
    if not p.exists():
        return Response({"detail": "No leaderboard yet. Run `make train`."},
                       status=status.HTTP_503_SERVICE_UNAVAILABLE)
    return Response({"leaderboard": json.loads(p.read_text())})


# ---------------------------------------------------------------------------
# Latest-month classification — what the classifier says for "now"
# ---------------------------------------------------------------------------

@api_view(["GET"])
def test_predictions(request):
    """Serve the held-out test set predictions for scatter / comparison plots."""
    p = METRICS / "test_predictions.csv"
    if not p.exists():
        return Response({"detail": "No test predictions. Run `make train`."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    df = pd.read_csv(p)
    cols = [c for c in ["station", "year", "month", "de_martonne", "pred_rf", "pred_xgb"] if c in df.columns]
    return Response({"predictions": df[cols].round(4).to_dict(orient="records")})


@api_view(["GET"])
def scoring(request):
    """Return full ML scoring report: per-class precision/recall/F1,
    extended leaderboard with bias + interval coverage, and model health."""
    results = {}

    lb_path = METRICS / "leaderboard.json"
    if lb_path.exists():
        results["leaderboard"] = json.loads(lb_path.read_text())

    cr_path = METRICS / "classifier_report.json"
    if cr_path.exists():
        results["classifier_report"] = json.loads(cr_path.read_text())

    health_path = METRICS / "model_health.json"
    if health_path.exists():
        results["model_health"] = json.loads(health_path.read_text())

    shap_path = METRICS / "shap_importance.csv"
    if shap_path.exists():
        import pandas as pd
        shap_df = pd.read_csv(shap_path)
        results["shap_importance"] = shap_df.head(10).to_dict(orient="records")

    if not results:
        return Response(
            {"detail": "No metrics found. Run `make train` first."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return Response(results)


@api_view(["GET"])
def latest_zone(request):
    """Return the classifier's prediction + probabilities for the latest
    observation of a given station. Exposes the AridityZoneClassifier output
    directly, for side-by-side comparison with the regressor."""
    station = request.query_params.get("station")
    if station not in VALID_STATIONS:
        return Response({"detail": f"station must be one of {VALID_STATIONS}"},
                       status=status.HTTP_400_BAD_REQUEST)

    try:
        from data_ingestion.features import build_features
        from data_ingestion.loaders import load_clean_csv
        from model_engine.inference import get_registry

        reg = get_registry()
        zclf = reg.get("zone")
        if zclf is None:
            return Response({"detail": "Classifier not trained. Run `make train`."},
                           status=status.HTTP_503_SERVICE_UNAVAILABLE)

        df = load_clean_csv(DATA_CSV)
        feat = build_features(df)
        nowcast_feats = [c for c in reg["feature_sets"]["nowcast"]
                        if c in feat.columns]
        rows = feat[feat["station"] == station].dropna(subset=nowcast_feats)
        if rows.empty:
            return Response({"detail": "No complete feature row."},
                           status=status.HTTP_503_SERVICE_UNAVAILABLE)
        row = rows.sort_values("date").tail(1)
        X = row[nowcast_feats].fillna(0)

        pred = zclf.predict(X)[0]
        probs = zclf.predict_proba(X)[0]
        class_probs = [
            {"zone": str(cls), "probability": float(p)}
            for cls, p in sorted(zip(zclf.classes_, probs),
                                 key=lambda x: -x[1])
        ]
        regressor_zone = str(row.iloc[0]["aridity_zone"])

        return Response({
            "station": station,
            "year": int(row.iloc[0]["year"]),
            "month": int(row.iloc[0]["month"]),
            "classifier_prediction": str(pred),
            "regressor_zone": regressor_zone,
            "agreement": str(pred) == regressor_zone,
            "class_probabilities": class_probs,
        })
    except Exception as exc:
        log.exception("latest_zone failed")
        return Response({"detail": str(exc)},
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)
