"""Versioned, reproducible candidate specifications and dataset lineage checks."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from invoiceops_ml.data import (
    DATASET_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURES,
    SPLIT_FILENAMES,
    TARGET,
)

CANDIDATE_SPECIFICATION_VERSION = "candidate-specification-v1"
DEFAULT_CANDIDATE_SPEC_PATH = Path("config/candidates/rf-candidate-v1.json")
SUPPORTED_MODEL_PARAMETERS = {"n_estimators": 200, "random_state": 202605, "n_jobs": -1}


@dataclass(frozen=True)
class CandidateSpec:
    """The complete immutable contract for one reproducible candidate."""

    specification_version: str
    name: str
    dataset_version: str
    dataset_seed: int
    dataset_rows: int
    feature_schema_version: str
    features: tuple[str, ...]
    target: str
    model_type: str
    model_parameters: dict[str, int]


def _invalid(message: str) -> ValueError:
    return ValueError(f"Invalid candidate specification: {message}")


def _require_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{name} must be an object")
    return value


def _require_keys(value: Mapping[str, object], name: str, keys: set[str]) -> None:
    if set(value) != keys:
        raise _invalid(f"{name} fields must be exactly: {', '.join(sorted(keys))}")


def _require_non_blank_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{name} must be a non-blank string")
    return value


def load_candidate_spec(path: Path = DEFAULT_CANDIDATE_SPEC_PATH) -> CandidateSpec:
    """Load a reviewed candidate specification and reject any unsupported contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _invalid(str(error)) from error
    root = _require_object(payload, "root")
    _require_keys(
        root, "root", {"specification_version", "name", "dataset", "feature_schema", "model"}
    )

    specification_version = _require_non_blank_string(
        root.get("specification_version"), "specification_version"
    )
    if specification_version != CANDIDATE_SPECIFICATION_VERSION:
        raise _invalid("specification_version is unsupported")
    name = _require_non_blank_string(root.get("name"), "name")

    dataset = _require_object(root.get("dataset"), "dataset")
    _require_keys(dataset, "dataset", {"version", "seed", "rows"})
    if dataset.get("version") != DATASET_VERSION:
        raise _invalid("dataset.version is incompatible")
    seed, rows = dataset.get("seed"), dataset.get("rows")
    if type(seed) is not int or type(rows) is not int or rows < 7:
        raise _invalid("dataset.seed and dataset.rows must be valid generator inputs")

    feature_schema = _require_object(root.get("feature_schema"), "feature_schema")
    _require_keys(feature_schema, "feature_schema", {"version", "features", "target"})
    if feature_schema.get("version") != FEATURE_SCHEMA_VERSION:
        raise _invalid("feature_schema.version is incompatible")
    features = feature_schema.get("features")
    if not isinstance(features, list) or tuple(features) != MODEL_FEATURES:
        raise _invalid("feature_schema.features is incompatible")
    if feature_schema.get("target") != TARGET:
        raise _invalid("feature_schema.target is incompatible")

    model = _require_object(root.get("model"), "model")
    _require_keys(model, "model", {"type", "parameters"})
    if model.get("type") != "random_forest":
        raise _invalid("model.type is unsupported")
    parameters = model.get("parameters")
    if parameters != SUPPORTED_MODEL_PARAMETERS:
        raise _invalid("model.parameters are unsupported")

    return CandidateSpec(
        specification_version=specification_version,
        name=name,
        dataset_version=DATASET_VERSION,
        dataset_seed=seed,
        dataset_rows=rows,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features=MODEL_FEATURES,
        target=TARGET,
        model_type="random_forest",
        model_parameters=dict(SUPPORTED_MODEL_PARAMETERS),
    )


def validate_materialized_dataset(spec: CandidateSpec, dataset_dir: Path) -> dict[str, str]:
    """Validate generated lineage metadata and calculate the split hashes used for this run."""
    metadata_path = dataset_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Candidate dataset metadata is unavailable or invalid: {error}"
        ) from error
    if not isinstance(metadata, Mapping):
        raise TypeError("Candidate dataset metadata must be an object")
    expected_metadata = {
        "dataset_version": spec.dataset_version,
        "feature_schema_version": spec.feature_schema_version,
        "seed": spec.dataset_seed,
        "rows": spec.dataset_rows,
        "target": spec.target,
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Candidate dataset metadata does not match the specification")

    metadata_hashes = metadata.get("split_sha256")
    if not isinstance(metadata_hashes, Mapping) or set(metadata_hashes) != set(SPLIT_FILENAMES):
        raise ValueError("Candidate dataset metadata split hashes are invalid")
    hashes = {}
    for filename in SPLIT_FILENAMES:
        try:
            digest = hashlib.sha256((dataset_dir / filename).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"Candidate dataset split is unavailable: {filename}") from error
        if metadata_hashes.get(filename) != digest:
            raise ValueError(f"Candidate dataset split hash does not match metadata: {filename}")
        hashes[filename] = digest
    return hashes
