from django.apps import AppConfig


class ModelEngineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "model_engine"
    verbose_name = "ML Model Engine"
