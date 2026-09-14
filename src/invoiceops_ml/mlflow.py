"""MLflow environment configuration shared by notebooks and Python code."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    """Connection settings read from the environment without exposing passwords."""

    tracking_uri: str
    tracking_username: str | None = None
    tracking_password: str | None = field(default=None, repr=False)
    workspace: str | None = None


def _environment_value(environ: Mapping[str, object], name: str) -> str | None:
    value = environ.get(name)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def mlflow_config_from_env(environ: Mapping[str, object] | None = None) -> MLflowConfig:
    """Return validated MLflow settings for either a local or remote server."""
    environment = os.environ if environ is None else environ
    tracking_uri = _environment_value(environment, "MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        message = "MLFLOW_TRACKING_URI must be set"
        raise ValueError(message)

    username = _environment_value(environment, "MLFLOW_TRACKING_USERNAME")
    password = _environment_value(environment, "MLFLOW_TRACKING_PASSWORD")
    if (username is None) != (password is None):
        message = "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set together"
        raise ValueError(message)

    return MLflowConfig(
        tracking_uri=tracking_uri,
        tracking_username=username,
        tracking_password=password,
        workspace=_environment_value(environment, "MLFLOW_WORKSPACE"),
    )


def tracking_uri_from_env(environ: Mapping[str, object] | None = None) -> str:
    """Return the configured MLflow tracking URI."""
    return mlflow_config_from_env(environ).tracking_uri


def configure_mlflow(config: MLflowConfig) -> None:
    """Configure the MLflow client with a validated tracking URI."""
    mlflow.set_tracking_uri(config.tracking_uri)
