import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from invoiceops_ml import gate
from invoiceops_ml.gate import (
    DEFAULT_GATE_CONFIG_PATH,
    GateResult,
    load_gate_config,
    run_quality_gate,
)


class StubMlflowClient:
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.run_ids: list[str] = []

    def get_run(self, run_id: str) -> SimpleNamespace:
        self.run_ids.append(run_id)
        return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics))


def write_gate_config(path: Path, **overrides: object) -> Path:
    config: dict[str, object] = {
        "version": "invoice-risk-gate-v1",
        "metrics": {
            "validation_recall": {"min": 0.18},
            "validation_precision": {"min": 0.48},
        },
    }
    config.update(overrides)
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_quality_gate_passes_an_explicit_run_that_meets_baseline_thresholds(
    tmp_path: Path,
) -> None:
    client = StubMlflowClient({"validation_recall": 0.18, "validation_precision": 0.48})

    result = run_quality_gate("run-pass", write_gate_config(tmp_path / "gate.json"), client)

    assert client.run_ids == ["run-pass"]
    assert result.passed is True
    assert result.version == "invoice-risk-gate-v1"
    assert result.thresholds == {"validation_recall": 0.18, "validation_precision": 0.48}
    assert result.metrics == {"validation_recall": 0.18, "validation_precision": 0.48}


def test_quality_gate_fails_an_explicit_run_that_misses_a_threshold(tmp_path: Path) -> None:
    client = StubMlflowClient({"validation_recall": 0.17, "validation_precision": 0.80})

    result = run_quality_gate("run-fail", write_gate_config(tmp_path / "gate.json"), client)

    assert result.passed is False
    assert result.checks == {"validation_recall": False, "validation_precision": True}


def test_quality_gate_rejects_runs_missing_required_validation_metrics(tmp_path: Path) -> None:
    client = StubMlflowClient({"validation_recall": 0.80})

    with pytest.raises(ValueError, match="Missing required MLflow metrics: validation_precision"):
        run_quality_gate("run-missing", write_gate_config(tmp_path / "gate.json"), client)


def test_default_gate_config_is_available_outside_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_gate_config(DEFAULT_GATE_CONFIG_PATH)

    assert config.version == "invoice-risk-gate-v1"
    assert config.thresholds == {"validation_recall": 0.18, "validation_precision": 0.48}


def test_wheel_contains_default_gate_config(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        check=True,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    wheel = next(tmp_path.glob("invoiceops_ml-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "invoiceops_ml/invoice-risk-gate-v1.json" in archive.namelist()


def test_cli_preserves_an_external_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_gate_config(tmp_path / "external-gate.json")
    received: dict[str, object] = {}

    def fake_run_quality_gate(run_id: str, config_path: Path) -> GateResult:
        received["run_id"] = run_id
        received["config_path"] = config_path
        return GateResult(
            run_id=run_id,
            version="invoice-risk-gate-v1",
            thresholds={},
            metrics={},
            checks={},
            passed=True,
        )

    monkeypatch.setattr(gate, "run_quality_gate", fake_run_quality_gate)

    assert gate.main(["--run-id", "run-external", "--config", str(config_path)]) == 0
    assert received == {"run_id": "run-external", "config_path": config_path}


@pytest.mark.parametrize(
    "config",
    [
        {"version": "", "metrics": {}},
        {
            "version": "invoice-risk-gate-v1",
            "metrics": {
                "validation_recall": {"min": 0.18},
                "validation_precision": {"min": 1.01},
            },
        },
    ],
)
def test_gate_config_rejects_invalid_values(tmp_path: Path, config: dict[str, object]) -> None:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid gate configuration"):
        load_gate_config(path)
