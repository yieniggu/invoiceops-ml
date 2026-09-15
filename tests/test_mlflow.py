from pathlib import Path

import mlflow
import pytest

from invoiceops_ml.mlflow import configure_mlflow, mlflow_config_from_env, tracking_uri_from_env
from invoiceops_ml.ownership import OwnershipContext, set_run_ownership_tags


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


def test_mlflow_config_from_env_supports_local_settings_without_credentials() -> None:
    config = mlflow_config_from_env({"MLFLOW_TRACKING_URI": "http://127.0.0.1:5000"})

    assert config.tracking_uri == "http://127.0.0.1:5000"
    assert config.tracking_username is None
    assert config.tracking_password is None
    assert config.workspace is None


def test_mlflow_config_from_env_supports_remote_credentials_and_workspace() -> None:
    config = mlflow_config_from_env(
        {
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": "student",
            "MLFLOW_TRACKING_PASSWORD": "not-a-real-password",
            "MLFLOW_WORKSPACE": "course-2027",
        }
    )

    assert config.tracking_uri == "https://mlflow.example"
    assert config.tracking_username == "student"
    assert config.tracking_password == "not-a-real-password"
    assert config.workspace == "course-2027"
    assert "not-a-real-password" not in repr(config)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"MLFLOW_TRACKING_URI": ""},
        {"MLFLOW_TRACKING_URI": False},
        {"MLFLOW_TRACKING_URI": 0},
    ],
)
def test_mlflow_config_from_env_rejects_missing_blank_or_falsy_tracking_uri(
    environment: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI must be set"):
        mlflow_config_from_env(environment)


@pytest.mark.parametrize(
    "environment",
    [
        {"MLFLOW_TRACKING_URI": "https://mlflow.example", "MLFLOW_TRACKING_USERNAME": "student"},
        {"MLFLOW_TRACKING_URI": "https://mlflow.example", "MLFLOW_TRACKING_PASSWORD": "secret"},
        {
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": "student",
            "MLFLOW_TRACKING_PASSWORD": False,
        },
    ],
)
def test_mlflow_config_from_env_rejects_incomplete_credentials(
    environment: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD must be set together",
    ):
        mlflow_config_from_env(environment)


def test_mlflow_config_from_env_treats_falsy_optional_values_as_unset() -> None:
    config = mlflow_config_from_env(
        {
            "MLFLOW_TRACKING_URI": "https://mlflow.example",
            "MLFLOW_TRACKING_USERNAME": False,
            "MLFLOW_TRACKING_PASSWORD": 0,
            "MLFLOW_WORKSPACE": " ",
        }
    )

    assert config.tracking_username is None
    assert config.tracking_password is None
    assert config.workspace is None


def test_configure_mlflow_sets_the_validated_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_uris: list[str] = []
    monkeypatch.setattr("invoiceops_ml.mlflow.mlflow.set_tracking_uri", configured_uris.append)

    configure_mlflow(mlflow_config_from_env({"MLFLOW_TRACKING_URI": "https://mlflow.example"}))

    assert configured_uris == ["https://mlflow.example"]


def test_set_run_ownership_tags_rejects_a_missing_active_run() -> None:
    tracking_uri = mlflow.get_tracking_uri()
    context = OwnershipContext(
        organization_slug="course-2027",
        owner_type="user",
        owner_id="ef14197c-8f5b-4aef-8fa7-310e4da998b7",
        created_by_rut="12.345.678-5",
    )

    try:
        assert mlflow.active_run() is None

        with pytest.raises(RuntimeError, match="requires an active MLflow run"):
            set_run_ownership_tags(context)

        assert mlflow.active_run() is None
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(tracking_uri)


@pytest.mark.parametrize(
    ("owner_type", "owner_id"),
    [
        ("user", "ef14197c-8f5b-4aef-8fa7-310e4da998b7"),
        ("group", "3515a7c6-baa4-44aa-a433-e7c52d79a57d"),
    ],
)
def test_set_run_ownership_tags_writes_metadata_to_the_existing_active_run(
    owner_type: str, owner_id: str, tmp_path: Path
) -> None:
    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    context = OwnershipContext(
        organization_slug="course-2027",
        owner_type=owner_type,
        owner_id=owner_id,
        created_by_rut="12.345.678-5",
    )

    try:
        with mlflow.start_run() as run:
            set_run_ownership_tags(context)

            tags = mlflow.get_run(run.info.run_id).data.tags
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(tracking_uri)

    assert {key: tags[key] for key in context.as_tags()} == context.as_tags()


@pytest.mark.parametrize(
    "context",
    [
        {"organization_slug": "", "owner_type": "user", "owner_id": "ef14197c-8f5b-4aef-8fa7-310e4da998b7", "created_by_rut": "1"},
        {"organization_slug": "course-2027", "owner_type": "team", "owner_id": "ef14197c-8f5b-4aef-8fa7-310e4da998b7", "created_by_rut": "1"},
        {"organization_slug": "course-2027", "owner_type": "user", "owner_id": " ", "created_by_rut": "1"},
        {"organization_slug": "course-2027", "owner_type": "group", "owner_id": "group-1", "created_by_rut": "1"},
    ],
)
def test_ownership_context_rejects_missing_metadata_or_an_unknown_owner_type(
    context: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        OwnershipContext(**context)


def test_ownership_context_rejects_a_non_uuid_owner_id() -> None:
    with pytest.raises(ValueError, match="owner_id must be an InvoiceOps UUID"):
        OwnershipContext(
            organization_slug="course-2027",
            owner_type="group",
            owner_id="data-science-group",
            created_by_rut="12345678-5",
        )
