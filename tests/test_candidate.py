import json
from pathlib import Path

import mlflow
import pytest

from invoiceops_ml.candidate import (
    DEFAULT_CANDIDATE_SPEC_PATH,
    CandidateSpec,
    load_candidate_spec,
    validate_materialized_dataset,
)
from invoiceops_ml.data import generate_synthetic_dataset
from invoiceops_ml.train import train_candidate


def write_spec(path: Path, **overrides: object) -> Path:
    specification: dict[str, object] = {
        "specification_version": "candidate-specification-v1",
        "name": "rf-candidate-v1",
        "dataset": {"version": "invoice-risk-v1", "seed": 202604, "rows": 12000},
        "feature_schema": {
            "version": "invoice-features-v1",
            "features": [
                "invoice_amount_cents",
                "vendor_tenure_days",
                "previous_incidents_12m",
                "amount_vs_vendor_median",
                "has_purchase_order",
                "three_way_match",
                "bank_account_recently_changed",
                "country_risk",
            ],
            "target": "manual_review_required",
        },
        "model": {
            "type": "random_forest",
            "parameters": {"n_estimators": 200, "random_state": 202605, "n_jobs": -1},
        },
    }
    specification.update(overrides)
    path.write_text(json.dumps(specification), encoding="utf-8")
    return path


def test_default_candidate_spec_is_valid_and_versioned() -> None:
    spec = load_candidate_spec(DEFAULT_CANDIDATE_SPEC_PATH)

    assert spec == CandidateSpec(
        specification_version="candidate-specification-v1",
        name="rf-candidate-v1",
        dataset_version="invoice-risk-v1",
        dataset_seed=202604,
        dataset_rows=12000,
        feature_schema_version="invoice-features-v1",
        features=(
            "invoice_amount_cents",
            "vendor_tenure_days",
            "previous_incidents_12m",
            "amount_vs_vendor_median",
            "has_purchase_order",
            "three_way_match",
            "bank_account_recently_changed",
            "country_risk",
        ),
        target="manual_review_required",
        model_type="random_forest",
        model_parameters={"n_estimators": 200, "random_state": 202605, "n_jobs": -1},
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        {},
        {
            "specification_version": "candidate-specification-v2",
            "name": "rf-candidate-v1",
            "dataset": {"version": "invoice-risk-v1", "seed": 202604, "rows": 12000},
            "feature_schema": {},
            "model": {},
        },
    ],
)
def test_loader_rejects_invalid_json_or_required_fields(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid candidate specification"):
        load_candidate_spec(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset": {"version": "another-dataset", "seed": 202604, "rows": 12000}},
        {
            "feature_schema": {
                "version": "wrong",
                "features": [],
                "target": "manual_review_required",
            }
        },
        {
            "feature_schema": {
                "version": "invoice-features-v1",
                "features": ["invoice_amount_cents"],
                "target": "another_target",
            }
        },
        {"model": {"type": "hist_gradient_boosting", "parameters": {}}},
        {
            "model": {
                "type": "random_forest",
                "parameters": {
                    "n_estimators": 200,
                    "random_state": 202605,
                    "n_jobs": -1,
                    "max_depth": 12,
                },
            }
        },
    ],
)
def test_loader_rejects_incompatible_contract_or_model_parameters(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="Invalid candidate specification"):
        load_candidate_spec(write_spec(tmp_path / "candidate.json", **overrides))


def test_materialized_dataset_validation_checks_metadata_and_real_split_hashes(
    tmp_path: Path,
) -> None:
    spec = load_candidate_spec(write_spec(tmp_path / "candidate.json"))
    dataset = generate_synthetic_dataset(seed=202604, rows=100, output_root=tmp_path / "data")

    with pytest.raises(ValueError, match="metadata"):
        validate_materialized_dataset(spec, dataset)

    dataset = generate_synthetic_dataset(seed=202604, rows=12000, output_root=tmp_path / "data")
    hashes = validate_materialized_dataset(spec, dataset)
    (dataset / "train.csv").write_text("tampered\n", encoding="utf-8")

    assert set(hashes) == {"train.csv", "validation.csv", "test.csv"}
    with pytest.raises(ValueError, match="hash"):
        validate_materialized_dataset(spec, dataset)


def test_training_materializes_validates_and_logs_the_candidate_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = write_spec(
        tmp_path / "candidate.json",
        dataset={"version": "invoice-risk-v1", "seed": 42, "rows": 100},
    )
    tracking_database = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tracking_database}")
    monkeypatch.setenv("INVOICEOPS_ORGANIZATION_SLUG", "course-2027")
    monkeypatch.setenv("INVOICEOPS_OWNER_TYPE", "user")
    monkeypatch.setenv("INVOICEOPS_OWNER_ID", "ef14197c-8f5b-4aef-8fa7-310e4da998b7")
    monkeypatch.setenv("INVOICEOPS_CREATED_BY_RUT", "12345678-5")
    selected_owners: list[object] = []
    monkeypatch.setattr("invoiceops_ml.train.select_owner_experiment", selected_owners.append)

    result = train_candidate(spec_path, tmp_path / "dataset" / "invoice-risk-v1")
    run = mlflow.MlflowClient().get_run(result.run_id)
    artifacts = mlflow.MlflowClient().list_artifacts(result.run_id)

    assert set(result.metrics) == {
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    }
    assert set(result.split_sha256) == {"train.csv", "validation.csv", "test.csv"}
    assert run.data.tags["candidate_name"] == "rf-candidate-v1"
    assert run.data.tags["dataset_split_sha256_train"] == result.split_sha256["train.csv"]
    assert run.data.params["n_estimators"] == "200"
    assert {artifact.path for artifact in artifacts} == {"candidate_specification", "model"}
    assert len(selected_owners) == 1
