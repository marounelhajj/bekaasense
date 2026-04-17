"""BekaaSense URL configuration."""
from django.contrib import admin
from django.urls import include, path
from django.http import JsonResponse


def healthcheck(request):
    """Liveness probe — used by Docker / monitoring."""
    return JsonResponse({"status": "ok", "service": "bekaasense"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", healthcheck, name="health"),
    path("api/", include("api.urls")),
    path("", include("dashboard.urls")),
]
