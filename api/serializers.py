"""DRF serializers — contract between API inputs/outputs and the engine."""
from rest_framework import serializers


VALID_STATIONS = ["Ammik", "Doures", "Ras_Baalbeck", "Tal_Amara"]


class ForecastRequestSerializer(serializers.Serializer):
    station = serializers.ChoiceField(choices=VALID_STATIONS)
    horizon_months = serializers.IntegerField(min_value=1, max_value=60, default=12)
    alpha = serializers.FloatField(min_value=0.01, max_value=0.5, default=0.1)


class ClassifyRequestSerializer(serializers.Serializer):
    station = serializers.ChoiceField(choices=VALID_STATIONS)
    year = serializers.IntegerField(min_value=1990, max_value=2100)
    month = serializers.IntegerField(min_value=1, max_value=12)


class ExplainRequestSerializer(serializers.Serializer):
    station = serializers.ChoiceField(choices=VALID_STATIONS)
    top_k = serializers.IntegerField(min_value=1, max_value=20, default=8)


class ForecastPointSerializer(serializers.Serializer):
    year = serializers.IntegerField()
    month = serializers.IntegerField()
    horizon = serializers.IntegerField()
    de_martonne_pred = serializers.FloatField()
    lower = serializers.FloatField()
    upper = serializers.FloatField()
    aridity_zone = serializers.CharField()
