"""MLflow environment configuration shared by notebooks and Python code."""

import os
from collections.abc import Mapping


def tracking_uri_from_env(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured MLflow tracking URI."""
    value = (os.environ if environ is None else environ).get("MLFLOW_TRACKING_URI", "").strip()
    if not value:
        message = "MLFLOW_TRACKING_URI must be set"
        raise ValueError(message)
    return value
