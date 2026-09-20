"""Versioned quality-gate evaluation for explicit MLflow runs."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Protocol

import mlflow

from invoiceops_ml.mlflow import mlflow_config_from_env

DEFAULT_GATE_CONFIG_PATH = files("invoiceops_ml").joinpath("invoice-risk-gate-v1.json")
REQUIRED_METRICS = ("validation_recall", "validation_precision")


class MLflowRunClient(Protocol):
    """The small MLflow client surface required by the quality gate."""

    def get_run(self, run_id: str) -> "MLflowRun": ...


class MLflowRunData(Protocol):
    """The metrics read from a completed MLflow run."""

    metrics: Mapping[str, float]


class MLflowRun(Protocol):
    """The completed MLflow run shape consumed by this gate."""

    data: MLflowRunData


@dataclass(frozen=True)
class GateConfig:
    """Validated immutable thresholds for a named quality-gate version."""

    version: str
    thresholds: dict[str, float]


@dataclass(frozen=True)
class GateResult:
    """The complete, serializable decision for one explicitly selected run."""

    run_id: str
    version: str
    thresholds: dict[str, float]
    metrics: dict[str, float]
    checks: dict[str, bool]
    passed: bool

    def as_report(self) -> dict[str, object]:
        """Return the stable JSON-compatible CLI report."""
        return asdict(self)


def _invalid_config(message: str) -> ValueError:
    return ValueError(f"Invalid gate configuration: {message}")


def _minimum(metrics: Mapping[str, object], metric_name: str) -> float:
    metric = metrics.get(metric_name)
    if not isinstance(metric, Mapping):
        raise _invalid_config(f"metrics.{metric_name} must be an object")
    minimum = metric.get("min")
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
        raise _invalid_config(f"metrics.{metric_name}.min must be a number")
    if not 0 <= minimum <= 1:
        raise _invalid_config(f"metrics.{metric_name}.min must be between 0 and 1")
    return float(minimum)


def load_gate_config(path: Traversable) -> GateConfig:
    """Load one versioned gate configuration from JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _invalid_config(str(error)) from error
    if not isinstance(payload, Mapping):
        raise _invalid_config("root must be an object")

    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise _invalid_config("version must be a non-blank string")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise _invalid_config("metrics must be an object")

    return GateConfig(
        version=version,
        thresholds={metric_name: _minimum(metrics, metric_name) for metric_name in REQUIRED_METRICS},
    )


def run_quality_gate(
    run_id: str,
    config_path: Traversable = DEFAULT_GATE_CONFIG_PATH,
    client: MLflowRunClient | None = None,
) -> GateResult:
    """Evaluate only the configured validation metrics for an explicit MLflow run."""
    if not run_id.strip():
        raise ValueError("run_id must not be blank")

    config = load_gate_config(config_path)
    mlflow_client = client
    if mlflow_client is None:
        tracking_uri = mlflow_config_from_env().tracking_uri
        mlflow_client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    run = mlflow_client.get_run(run_id)
    run_metrics = run.data.metrics
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in run_metrics]
    if missing_metrics:
        raise ValueError(f"Missing required MLflow metrics: {', '.join(missing_metrics)}")

    metrics = {metric: float(run_metrics[metric]) for metric in REQUIRED_METRICS}
    checks = {metric: metrics[metric] >= config.thresholds[metric] for metric in REQUIRED_METRICS}
    return GateResult(
        run_id=run_id,
        version=config.version,
        thresholds=config.thresholds,
        metrics=metrics,
        checks=checks,
        passed=all(checks.values()),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the quality gate and return a process status for automation callers."""
    parser = argparse.ArgumentParser(description="Evaluate an explicit MLflow run with a versioned gate.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a versioned gate JSON configuration",
    )
    args = parser.parse_args(argv)

    result = run_quality_gate(args.run_id, args.config or DEFAULT_GATE_CONFIG_PATH)
    print(json.dumps(result.as_report(), sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
