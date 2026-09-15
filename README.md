# InvoiceOps ML

Workspace principal de investigación para los alumnos de InvoiceOps. Los
notebooks explican el flujo didáctico y el código reusable vive en `src/`.

## Alcance actual

Este repositorio incluye la configuración reusable de MLflow de ML-01 y los tags
de ownership de ML-02. No incluye notebooks, datasets, entrenamiento, Docker ni
infraestructura de MLflow.

## Requisitos

- Python `3.12`.
- [uv](https://docs.astral.sh/uv/).

## Instalación local

Desde la raíz del repositorio:

```bash
uv sync --all-groups
cp .env.example .env
uv run python -m ipykernel install --user --name invoiceops-ml-py312 \
  --display-name "InvoiceOps ML Python 3.12"
```

El archivo `.env` es local y está ignorado por Git. No incluya credenciales ni
otros secretos en notebooks, código, salidas o commits.

## Configuración de MLflow

Los notebooks y el código reusable leen la misma configuración desde variables de
entorno. No escriba URLs ni credenciales en notebooks. La función
`invoiceops_ml.mlflow.mlflow_config_from_env()` la valida y
`configure_mlflow()` configura la URI del cliente.

### MLflow local

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

No defina credenciales si el servidor local no las exige. El stack local vive en
`../invoiceops-mlflow`.

### MLflow remoto con autenticación y Workspace

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.edu
export MLFLOW_TRACKING_USERNAME=student@example.edu
export MLFLOW_TRACKING_PASSWORD='obtain-this-value-from-the-approved-secret-store'
export MLFLOW_WORKSPACE=course-2027
```

`MLFLOW_TRACKING_USERNAME` y `MLFLOW_TRACKING_PASSWORD` deben definirse juntos.
No los imprima, persista, agregue a notebooks, salidas ni commits. `MLFLOW_WORKSPACE`
es opcional y selecciona el Workspace activo cuando el servidor lo requiere.

En ambos casos, el notebook sólo obtiene y aplica la configuración reusable:

```python
from invoiceops_ml.mlflow import configure_mlflow, mlflow_config_from_env

configure_mlflow(mlflow_config_from_env())
```

## Ownership metadata

Cada run debe incluir el contexto académico que permite encontrarlo en la UI de
MLflow por organización y propietario. El contrato de tags es estable:

```text
organization_slug
owner_type = user | group
owner_id
created_by_rut
```

Use el mismo contexto reutilizable para trabajo individual y grupal, dentro de
una ejecución activa. `owner_id` es el UUID estable proporcionado por
InvoiceOps: `User.id` para trabajo individual y `Group.id` para trabajo grupal.
No lo transforme ni lo derive a partir de nombres o slugs en el notebook.
`created_by_rut` es el RUT normalizado del creador.

```python
import mlflow

from invoiceops_ml.ownership import OwnershipContext, set_run_ownership_tags

with mlflow.start_run():
    set_run_ownership_tags(
        OwnershipContext(
            organization_slug="course-2027",
            owner_type="group",
            owner_id="3515a7c6-baa4-44aa-a433-e7c52d79a57d",
            created_by_rut="12345678-5",
        )
    )
```

Los tags permanecen visibles y se pueden filtrar en la UI de MLflow. El
contrato completo de INT-02 está en
`../dev/tickets/INT-02_ownership_academico_mlflow.md`. Las convenciones de
nomenclatura de Experiment y Registered Model corresponden a MLFLOW-05.

## Estructura

```text
notebooks/          Material didáctico
src/invoiceops_ml/  Código Python reusable
config/             Configuración versionada no secreta
tests/              Pruebas del código reusable
.github/workflows/  Automatización futura
```

## Verificación

```bash
uv run pytest
uv run ruff check .
```

No ejecute servicios desde este repositorio. La selección local o remota ocurre
exclusivamente mediante variables de entorno.
