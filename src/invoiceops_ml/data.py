"""Deterministic synthetic dataset generation for InvoiceOps ML lessons."""

import csv
import hashlib
import json
import math
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

DATASET_VERSION = "invoice-risk-v1"
FEATURE_SCHEMA_VERSION = "invoice-features-v1"
MODEL_FEATURES = (
    "invoice_amount_cents",
    "vendor_tenure_days",
    "previous_incidents_12m",
    "amount_vs_vendor_median",
    "has_purchase_order",
    "three_way_match",
    "bank_account_recently_changed",
    "country_risk",
)
SPLIT_FILENAMES = ("train.csv", "validation.csv", "test.csv")
TARGET = "manual_review_required"
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def seed_from_rut(rut: object) -> int:
    """Return a stable seed for an already-normalized InvoiceOps RUT."""
    if not isinstance(rut, str) or not rut.strip():
        raise ValueError("rut must be a non-empty normalized InvoiceOps RUT")
    return int.from_bytes(hashlib.sha256(rut.encode("utf-8")).digest()[:8], "big")


def _validate_version(version: str) -> None:
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError("version must be a safe dataset directory name")


def _validate_rows(rows: int) -> None:
    if type(rows) is not int or rows < 7:
        raise ValueError("rows must be an integer of at least 7")


def _risk_label(record: dict[str, object], rng: random.Random) -> bool:
    score = -1.0 + 0.4 * record["previous_incidents_12m"] + rng.gauss(0.0, 0.35)
    score += 0.9 if record["bank_account_recently_changed"] else 0.0
    score += {"low": 0.0, "medium": 0.25, "high": 0.75}[record["country_risk"]]
    score += 0.75 * max(0.0, record["amount_vs_vendor_median"] - 1.25)
    score += 0.6 if record["vendor_tenure_days"] < 180 else 0.0
    score += 0.5 if record["invoice_amount_cents"] > 1_000_000 else 0.0
    score -= 0.45 if record["has_purchase_order"] else 0.0
    score -= 0.5 if record["three_way_match"] else 0.0
    score -= 0.3 if record["vendor_tenure_days"] > 2_000 else 0.0
    score += 0.75 if record["vendor_tenure_days"] < 180 and record["bank_account_recently_changed"] else 0.0
    score += 0.7 if record["country_risk"] == "high" and record["amount_vs_vendor_median"] > 1.5 else 0.0
    return rng.random() < 1.0 / (1.0 + math.exp(-score))


def _generate_rows(rows: int, rng: random.Random) -> list[dict[str, object]]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    generated = []
    for index in range(rows):
        tenure = rng.randint(0, 180) if rng.random() < 0.15 else int(180 + 3_470 * (1 - rng.random() ** 2))
        incident_roll = rng.random()
        incidents = 0 if incident_roll < 0.78 else 1 if incident_roll < 0.93 else rng.randint(2, 5)
        record: dict[str, object] = {
            "invoice_id": f"INV-SYN-{index + 1:06d}",
            "submitted_at": (start + timedelta(hours=index, minutes=rng.randint(0, 59))).isoformat(),
            "invoice_amount_cents": max(1_000, min(5_000_000, round(math.exp(rng.gauss(math.log(180_000), 0.9))))),
            "vendor_tenure_days": tenure,
            "previous_incidents_12m": incidents,
            "amount_vs_vendor_median": round(math.exp(rng.gauss(-0.06125, 0.35)), 6),
            "has_purchase_order": rng.random() < 0.85,
            "three_way_match": rng.random() < 0.8,
            "bank_account_recently_changed": rng.random() < 0.1,
            "country_risk": rng.choices(("low", "medium", "high"), weights=(65, 25, 10))[0],
        }
        record[TARGET] = _risk_label(record, rng)
        generated.append(record)
    return generated


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["invoice_id", "submitted_at", *MODEL_FEATURES, TARGET], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_synthetic_dataset(
    *, seed: int, rows: int, version: str = DATASET_VERSION, output_root: Path = Path("data")
) -> Path:
    """Generate a deterministic, chronologically split invoice-risk dataset."""
    _validate_version(version)
    _validate_rows(rows)

    dataset_dir = output_root / version
    dataset_dir.mkdir(parents=True, exist_ok=True)
    generated = _generate_rows(rows, random.Random(seed))
    train_end = rows * 70 // 100
    validation_end = train_end + rows * 15 // 100
    splits = {
        "train.csv": generated[:train_end],
        "validation.csv": generated[train_end:validation_end],
        "test.csv": generated[validation_end:],
    }
    for filename in SPLIT_FILENAMES:
        _write_split(dataset_dir / filename, splits[filename])

    metadata = {
        "dataset_version": version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "rows": rows,
        "seed": seed,
        "split_sha256": {
            filename: hashlib.sha256((dataset_dir / filename).read_bytes()).hexdigest()
            for filename in SPLIT_FILENAMES
        },
        "target": TARGET,
        "test_rows": len(splits["test.csv"]),
        "train_rows": len(splits["train.csv"]),
        "validation_rows": len(splits["validation.csv"]),
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset_dir
