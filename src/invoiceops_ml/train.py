"""Notebook-free training for reviewed InvoiceOps candidate specifications."""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from invoiceops_ml.candidate import (
    CandidateSpec,
    load_candidate_spec,
    validate_materialized_dataset,
)
from invoiceops_ml.data import generate_synthetic_dataset
from invoiceops_ml.mlflow import configure_mlflow, mlflow_config_from_env
from invoiceops_ml.ownership import (
    OwnershipContext,
    select_owner_experiment,
    set_run_ownership_tags,
)

DEFAULT_DATASET_DIR = Path("notebooks/data/invoice-risk-v1")


@dataclass(frozen=True)
class TrainingResult:
    """The stable report emitted after one candidate run."""

    run_id: str
    candidate_name: str
    dataset_dir: str
    split_sha256: dict[str, str]
    metrics: dict[str, float]

    def as_report(self) -> dict[str, object]:
        return asdict(self)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set from the current InvoiceOps context")
    return value


def _ownership_context_from_env() -> OwnershipContext:
    return OwnershipContext(
        organization_slug=_required_environment("INVOICEOPS_ORGANIZATION_SLUG"),
        owner_type=_required_environment("INVOICEOPS_OWNER_TYPE"),
        owner_id=_required_environment("INVOICEOPS_OWNER_ID"),
        created_by_rut=_required_environment("INVOICEOPS_CREATED_BY_RUT"),
    )


def _load_split(dataset_dir: Path, split: str, target: str) -> tuple[list[list[object]], list[int]]:
    with (dataset_dir / f"{split}.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    features = [
        [
            int(row["invoice_amount_cents"]),
            int(row["vendor_tenure_days"]),
            int(row["previous_incidents_12m"]),
            float(row["amount_vs_vendor_median"]),
            int(row["has_purchase_order"] == "True"),
            int(row["three_way_match"] == "True"),
            int(row["bank_account_recently_changed"] == "True"),
            row["country_risk"],
        ]
        for row in rows
    ]
    return features, [int(row[target] == "True") for row in rows]


def _build_pipeline(spec: CandidateSpec) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", tuple(range(7))),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), (7,)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**spec.model_parameters)),
        ]
    )


def _classification_metrics(
    pipeline: Pipeline, prefix: str, features: list[list[object]], target: list[int]
) -> dict[str, float]:
    predictions = pipeline.predict(features)
    return {
        f"{prefix}_accuracy": accuracy_score(target, predictions),
        f"{prefix}_precision": precision_score(target, predictions, zero_division=0),
        f"{prefix}_recall": recall_score(target, predictions, zero_division=0),
        f"{prefix}_f1": f1_score(target, predictions, zero_division=0),
    }


def train_candidate(spec_path: Path, dataset_dir: Path = DEFAULT_DATASET_DIR) -> TrainingResult:
    """Materialize, validate, train, evaluate, and log one reviewed candidate specification."""
    spec = load_candidate_spec(spec_path)
    if dataset_dir.name != spec.dataset_version:
        raise ValueError("dataset_dir must end with the candidate dataset version")
    materialized_dataset = generate_synthetic_dataset(
        seed=spec.dataset_seed, rows=spec.dataset_rows, output_root=dataset_dir.parent
    )
    split_hashes = validate_materialized_dataset(spec, materialized_dataset)
    train_features, train_target = _load_split(materialized_dataset, "train", spec.target)
    validation_features, validation_target = _load_split(
        materialized_dataset, "validation", spec.target
    )
    test_features, test_target = _load_split(materialized_dataset, "test", spec.target)

    pipeline = _build_pipeline(spec)
    pipeline.fit(train_features, train_target)
    metrics = {
        **_classification_metrics(pipeline, "validation", validation_features, validation_target),
        **_classification_metrics(pipeline, "test", test_features, test_target),
    }

    configure_mlflow(mlflow_config_from_env())
    ownership_context = _ownership_context_from_env()
    select_owner_experiment(ownership_context)
    with mlflow.start_run(run_name=spec.name) as run:
        set_run_ownership_tags(ownership_context)
        mlflow.set_tags(
            {
                "candidate_name": spec.name,
                "candidate_specification_version": spec.specification_version,
                "dataset_version": spec.dataset_version,
                **{
                    f"dataset_split_sha256_{name.removesuffix('.csv')}": digest
                    for name, digest in split_hashes.items()
                },
            }
        )
        mlflow.log_params({"candidate_model_type": spec.model_type, **spec.model_parameters})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(spec_path), artifact_path="candidate_specification")
        with TemporaryDirectory() as model_dir:
            mlflow.sklearn.save_model(pipeline, model_dir)
            mlflow.log_artifacts(model_dir, artifact_path="model")
        run_id = run.info.run_id
    return TrainingResult(
        run_id=run_id,
        candidate_name=spec.name,
        dataset_dir=str(materialized_dataset),
        split_sha256=split_hashes,
        metrics=metrics,
    )


def main(argv: list[str] | None = None) -> int:
    """Train the requested candidate and write its run report as JSON."""
    parser = argparse.ArgumentParser(
        description="Train a versioned InvoiceOps candidate specification."
    )
    parser.add_argument(
        "--spec", type=Path, required=True, help="Path to a candidate specification JSON file"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory where the reproducible dataset will be materialized",
    )
    args = parser.parse_args(argv)
    print(json.dumps(train_candidate(args.spec, args.dataset_dir).as_report(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
