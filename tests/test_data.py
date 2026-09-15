import csv
import hashlib
import json
from pathlib import Path

import pytest

from invoiceops_ml.data import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURES,
    SPLIT_FILENAMES,
    TARGET,
    generate_synthetic_dataset,
    seed_from_rut,
)


def _read_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_dataset_generation_is_reproducible_and_records_lineage(tmp_path: Path) -> None:
    first = generate_synthetic_dataset(seed=42, rows=100, output_root=tmp_path / "first")
    second = generate_synthetic_dataset(seed=42, rows=100, output_root=tmp_path / "second")

    for filename in (*SPLIT_FILENAMES, "metadata.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_version"] == "invoice-risk-v1"
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["seed"] == 42
    assert metadata["target"] == TARGET
    assert metadata["split_sha256"] == {
        filename: hashlib.sha256((first / filename).read_bytes()).hexdigest()
        for filename in SPLIT_FILENAMES
    }


def test_dataset_generation_creates_chronological_train_validation_test_splits(tmp_path: Path) -> None:
    dataset = generate_synthetic_dataset(seed=91, rows=101, output_root=tmp_path)
    train, validation, test = (_read_split(dataset / filename) for filename in SPLIT_FILENAMES)

    assert [len(split) for split in (train, validation, test)] == [70, 15, 16]
    assert list(train[0]) == ["invoice_id", "submitted_at", *MODEL_FEATURES, TARGET]
    assert train[-1]["submitted_at"] < validation[0]["submitted_at"] < test[0]["submitted_at"]
    assert len({row["invoice_id"] for split in (train, validation, test) for row in split}) == 101


def test_seed_from_rut_is_stable_and_distinguishes_normalized_ruts() -> None:
    assert seed_from_rut("12345678-5") == seed_from_rut("12345678-5")
    assert seed_from_rut("12345678-5") != seed_from_rut("87654321-4")


@pytest.mark.parametrize("rut", ["", " ", None])
def test_seed_from_rut_rejects_missing_values(rut: str | None) -> None:
    with pytest.raises(ValueError, match="rut"):
        seed_from_rut(rut)


@pytest.mark.parametrize("rows", [0, -1, True, 6])
def test_dataset_generation_rejects_invalid_row_counts(tmp_path: Path, rows: int) -> None:
    with pytest.raises(ValueError, match="rows"):
        generate_synthetic_dataset(seed=1, rows=rows, output_root=tmp_path)


@pytest.mark.parametrize("version", ["../outside", "/absolute", "nested/version", ""])
def test_dataset_generation_rejects_unsafe_versions(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="version"):
        generate_synthetic_dataset(seed=1, rows=10, version=version, output_root=tmp_path)
