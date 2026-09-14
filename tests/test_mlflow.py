import pytest

from invoiceops_ml.mlflow import tracking_uri_from_env


def test_tracking_uri_from_env_reads_the_environment_value() -> None:
    environment = {"MLFLOW_TRACKING_URI": "https://mlflow.example"}

    assert tracking_uri_from_env(environment) == "https://mlflow.example"


@pytest.mark.parametrize(
    "environment",
    [{}, {"MLFLOW_TRACKING_URI": " \t"}],
)
def test_tracking_uri_from_env_rejects_a_missing_or_blank_value(
    environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://process.example")

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI must be set"):
        tracking_uri_from_env(environment)


def test_tracking_uri_from_env_uses_process_environment_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://process.example")

    assert tracking_uri_from_env() == "https://process.example"
