"""Academic ownership metadata shared by MLflow consumers."""

from dataclasses import dataclass
from typing import Literal

import mlflow

OwnerType = Literal["user", "group"]


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

    def as_tags(self) -> dict[str, str]:
        """Return the stable tag contract used by runs and registered models."""
        return {
            "organization_slug": self.organization_slug,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "created_by_rut": self.created_by_rut,
        }


def set_run_ownership_tags(context: OwnershipContext) -> None:
    """Write ownership tags to the current MLflow run for UI filtering."""
    if mlflow.active_run() is None:
        raise RuntimeError("set_run_ownership_tags requires an active MLflow run")
    mlflow.set_tags(context.as_tags())
