"""Opt-in, loopback-only promotion for a disposable InvoiceOps demonstration stack."""

import argparse
import ipaddress
import json
import math
import os
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

import mlflow

from invoiceops_ml.gate import GateResult, run_quality_gate
from invoiceops_ml.ownership import (
    PRODUCTION_REGISTERED_MODEL_NAME,
    OwnershipContext,
    owner_experiment_name,
)

LOCAL_CANDIDATE_NAMES = frozenset({"hist-gradient-boosting"})
DECISION_METRICS = ("validation_recall", "validation_precision", "validation_f1")
LOCAL_GATE_VERSION = "invoice-risk-gate-v1"


def _local_context(environment: Mapping[str, str]) -> tuple[str, OwnershipContext]:
    uri = environment.get("MLFLOW_TRACKING_URI", "")
    parsed = urlsplit(uri)
    hostname = parsed.hostname
    try:
        local = hostname == "localhost" or (
            hostname is not None and ipaddress.ip_address(hostname).is_loopback
        )
        port = parsed.port
    except ValueError:
        local = False
        port = None
    if (
        parsed.scheme != "http"
        or not local
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Promotion requires an HTTP loopback MLflow tracking URI with a port")
    if environment.get("MLFLOW_WORKSPACE", "").strip() not in {"", "default"}:
        raise ValueError("Local promotion requires the default MLflow workspace")
    if environment.get("MLFLOW_REGISTRY_URI", "").strip() not in {"", uri}:
        raise ValueError("MLFLOW_REGISTRY_URI must match the loopback tracking URI")

    def required(name: str) -> str:
        value = environment.get(name, "").strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value

    context = OwnershipContext(
        organization_slug=required("INVOICEOPS_ORGANIZATION_SLUG"),
        owner_type=required("INVOICEOPS_OWNER_TYPE"),
        owner_id=required("INVOICEOPS_OWNER_ID"),
        created_by_rut=required("INVOICEOPS_CREATED_BY_RUT"),
    )
    return uri, context


def promote_local_run(
    run_id: str,
    environment: Mapping[str, str] | None = None,
    client: object | None = None,
    gate: Callable[[str, object], GateResult] = lambda run_id, client: run_quality_gate(
        run_id, client=client
    ),
    register: Callable[[str, str], object] = mlflow.register_model,
) -> str:
    """Check ownership, artifact and versioned Gate before changing the local alias."""
    if not run_id.strip() or "/" in run_id or "\\" in run_id:
        raise ValueError("An explicit run ID is required")
    uri, context = _local_context(os.environ if environment is None else environment)
    model_client = (
        mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri) if client is None else client
    )
    run = model_client.get_run(run_id)
    experiment = model_client.get_experiment_by_name(owner_experiment_name(context))
    if experiment is None or not _eligible_run(run, experiment.experiment_id, context):
        raise ValueError("Run is not an eligible loader-compatible local candidate")
    if not any(
        artifact.path == "model" and artifact.is_dir
        for artifact in model_client.list_artifacts(run_id)
    ):
        raise ValueError("Run has no model artifact")
    decision = gate(run_id, model_client)
    if decision.run_id != run_id or decision.version != LOCAL_GATE_VERSION or not decision.passed:
        raise ValueError("Quality Gate FAIL or run mismatch; no promotion performed")

    versions = model_client.search_model_versions(f"name='{PRODUCTION_REGISTERED_MODEL_NAME}'")
    existing = next((version for version in versions if version.run_id == run_id), None)
    if existing is None:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        existing = register(f"runs:/{run_id}/model", PRODUCTION_REGISTERED_MODEL_NAME)
        if existing.run_id != run_id:
            raise ValueError("Registered model version does not match the selected run")
    model_client.set_registered_model_alias(
        PRODUCTION_REGISTERED_MODEL_NAME, "champion", existing.version
    )
    return str(existing.version)


def _eligible_run(run: object, experiment_id: str, context: OwnershipContext) -> bool:
    tags = run.data.tags
    name = tags.get("mlflow.runName")
    metrics = run.data.metrics
    return (
        run.info.experiment_id == experiment_id
        and run.info.status == "FINISHED"
        and all(tags.get(key) == value for key, value in context.as_tags().items())
        and name in LOCAL_CANDIDATE_NAMES
        and (tags.get("candidate_name") is None or tags["candidate_name"] == name)
        and all(
            key in metrics and math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1
            for key in DECISION_METRICS
        )
    )


def select_and_promote_local(
    run_ids: list[str],
    environment: Mapping[str, str] | None = None,
    client: object | None = None,
    gate: Callable[[str, object], GateResult] = lambda run_id, client: run_quality_gate(
        run_id, client=client
    ),
    register: Callable[[str, str], object] = mlflow.register_model,
) -> dict[str, object]:
    """Select the best Gate-PASS run from an explicit, bounded local owner run list."""
    if not run_ids or any(not run_id.strip() or "/" in run_id or "\\" in run_id for run_id in run_ids):
        raise ValueError("At least one valid candidate run ID is required")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Candidate run IDs must be unique")
    uri, context = _local_context(os.environ if environment is None else environment)
    model_client = (
        mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri) if client is None else client
    )
    experiment = model_client.get_experiment_by_name(owner_experiment_name(context))
    if experiment is None:
        raise ValueError("Owner experiment does not exist; no promotion performed")

    passing = []
    for run_id in run_ids:
        run = model_client.get_run(run_id)
        if not _eligible_run(run, experiment.experiment_id, context):
            continue
        metrics = run.data.metrics
        if not any(
            artifact.path == "model" and artifact.is_dir
            for artifact in model_client.list_artifacts(run_id)
        ):
            continue
        decision = gate(run_id, model_client)
        if decision.run_id != run_id or decision.version != LOCAL_GATE_VERSION:
            raise ValueError("Quality Gate run or version mismatch; no promotion performed")
        if decision.passed:
            passing.append((run_id, tuple(metrics[key] for key in DECISION_METRICS)))

    if not passing:
        raise ValueError("No eligible Gate-PASS candidates; no promotion performed")
    # Descending validation-only scores; ascending run ID makes exact ties reproducible.
    winner, _ = min(passing, key=lambda item: (*(-score for score in item[1]), item[0]))
    final_decision = None

    def recheck(run_id: str, checked_client: object) -> GateResult:
        nonlocal final_decision
        final_decision = gate(run_id, checked_client)
        return final_decision

    version = promote_local_run(winner, environment, model_client, recheck, register)
    return {
        "run_id": winner,
        "model": PRODUCTION_REGISTERED_MODEL_NAME,
        "version": version,
        "alias": "champion",
        "gate_metrics": final_decision.metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a Gate-passing run in local MLflow only")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", help="Explicit run ID for backward-compatible promotion")
    selection.add_argument(
        "--candidate-run-id", action="append", dest="candidate_run_ids",
        help="Select from these explicit owner run IDs (repeat for each candidate)",
    )
    args = parser.parse_args(argv)
    if args.candidate_run_ids:
        report = select_and_promote_local(args.candidate_run_ids)
    else:
        decision = None

        def checked_gate(run_id: str, client: object) -> GateResult:
            nonlocal decision
            decision = run_quality_gate(run_id, client=client)
            return decision

        version = promote_local_run(args.run_id, gate=checked_gate)
        report = {
            "run_id": args.run_id, "model": PRODUCTION_REGISTERED_MODEL_NAME,
            "version": version, "alias": "champion", "gate_metrics": decision.metrics,
        }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
