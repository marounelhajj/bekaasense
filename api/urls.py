"""API URL routing."""
from django.urls import path
from api import views

urlpatterns = [
    path("stations/", views.stations, name="stations"),
    path("predict/", views.predict, name="predict"),
    path("classify/", views.classify, name="classify"),
    path("trend/", views.trend, name="trend"),
    path("explain/", views.explain, name="explain"),
    path("leaderboard/", views.leaderboard, name="leaderboard"),
    path("scoring/", views.scoring, name="scoring"),
    path("test_predictions/", views.test_predictions, name="test_predictions"),
    path("latest_zone/", views.latest_zone, name="latest_zone"),
]
