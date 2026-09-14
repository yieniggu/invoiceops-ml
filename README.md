# InvoiceOps ML

Workspace principal de investigación para los alumnos de InvoiceOps. Los
notebooks explican el flujo didáctico y el código reusable vive en `src/`.

## Alcance actual

Este repositorio contiene solamente el scaffold inicial de ML-00. No incluye
notebooks, datasets, entrenamiento, Docker ni infraestructura de MLflow.

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

La URI de tracking se lee desde `MLFLOW_TRACKING_URI`:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

`invoiceops_ml.mlflow.tracking_uri_from_env()` entrega esa configuración al
código reusable. ML-01 definirá los perfiles local/remoto y las variables de
autenticación y Workspace; los notebooks no deben contener URLs ni credenciales.

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

No ejecute servicios desde este repositorio en ML-00. El stack local de MLflow
vive en `../invoiceops-mlflow` y se consume mediante la variable de entorno.
