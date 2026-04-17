"""Dashboard views — renders the single-page Chart.js UI."""
from django.shortcuts import render


def index(request):
    return render(request, "dashboard/index.html")
