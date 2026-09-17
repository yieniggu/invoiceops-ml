"""Academic ownership metadata shared by MLflow consumers."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import mlflow

OwnerType = Literal["user", "group"]

INVOICE_RISK_EXPERIMENT = "invoice-risk"
INVOICE_REVIEW_MODEL = "invoice-review"
PRODUCTION_REGISTERED_MODEL_NAME = "invoice-review-production"


@dataclass(frozen=True)
class OwnershipContext:
    """Metadata that identifies the academic owner of an MLflow resource."""

    organization_slug: str
    owner_type: OwnerType
    owner_id: str
    created_by_rut: str

    def __post_init__(self) -> None:
        values = (self.organization_slug, self.owner_id, self.created_by_rut)
        if any(not value.strip() for value in values):
            raise ValueError("Ownership metadata values must not be blank")
        if self.owner_type not in {"user", "group"}:
            raise ValueError("owner_type must be 'user' or 'group'")
        try:
            UUID(self.owner_id)
        except ValueError as error:
            raise ValueError("owner_id must be an InvoiceOps UUID") from error

    def as_tags(self) -> dict[str, str]:
        """Return the stable tag contract used by runs and registered models."""
        return {
            "organization_slug": self.organization_slug,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "created_by_rut": self.created_by_rut,
        }


def owner_experiment_name(context: OwnershipContext) -> str:
    """Return the canonical experiment name for an individual or group owner."""
    if context.owner_type == "user":
        return f"student/{context.created_by_rut}/{INVOICE_RISK_EXPERIMENT}"
    return f"group/{context.owner_id}/{INVOICE_RISK_EXPERIMENT}"


def owner_registered_model_name(context: OwnershipContext) -> str:
    """Return the canonical registered model name for an individual or group owner."""
    if context.owner_type == "user":
        return f"student-{context.created_by_rut}-{INVOICE_REVIEW_MODEL}"
    return f"group-{context.owner_id}-{INVOICE_REVIEW_MODEL}"


def select_owner_experiment(context: OwnershipContext) -> None:
    """Select the owner's experiment before opening an MLflow run."""
    mlflow.set_experiment(owner_experiment_name(context))


def set_run_ownership_tags(context: OwnershipContext) -> None:
    """Write ownership tags to the current MLflow run for UI filtering."""
    if mlflow.active_run() is None:
        raise RuntimeError("set_run_ownership_tags requires an active MLflow run")
    mlflow.set_tags(context.as_tags())


def set_registered_model_ownership_tags(
    registered_model_name: str, context: OwnershipContext, client: mlflow.MlflowClient | None = None
) -> None:
    """Write the stable ownership tag contract to an owner-scoped registered model."""
    if registered_model_name == PRODUCTION_REGISTERED_MODEL_NAME:
        raise ValueError("Cannot write academic ownership tags to the shared production model")
    model_client = mlflow.MlflowClient() if client is None else client
    for key, value in context.as_tags().items():
        model_client.set_registered_model_tag(registered_model_name, key, value)
