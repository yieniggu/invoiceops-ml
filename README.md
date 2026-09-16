# InvoiceOps ML

Workspace principal de investigación para los alumnos de InvoiceOps. Los
notebooks explican el flujo didáctico y el código reusable vive en `src/`.

## Alcance actual

Este repositorio incluye la configuración reusable de MLflow de ML-01, los tags
de ownership de ML-02, el generador de datasets de ML-03 y los notebooks
didácticos de ML-04, ML-05, ML-06 y ML-07. No incluye Docker ni infraestructura de
MLflow.

## Requisitos

- Python `3.12`.
- [uv](https://docs.astral.sh/uv/).

## Instalación local

Desde la raíz del repositorio:

```bash
uv sync --all-groups
./scripts/register-kernel.sh
```

`uv sync --all-groups` crea o actualiza `.venv` con las dependencias del
proyecto, incluido el grupo `teaching` con Jupyter e `ipykernel`. No registra un
kernel de Jupyter. `./scripts/register-kernel.sh` registra
`invoiceops-ml-py312` en `.venv` mediante `--sys-prefix`; es seguro repetirlo y
no requiere instalación global, privilegios ni `--user`.

Verifique que Jupyter iniciado desde el entorno del proyecto lo descubre:

```bash
uv run --group teaching jupyter kernelspec list
uv run --group teaching jupyter kernelspec list --json
```

La segunda orden debe mostrar `invoiceops-ml-py312` bajo
`.venv/share/jupyter/kernels/` y su `argv` debe comenzar con el intérprete de
`.venv`. Para abrir los notebooks con ese entorno:

```bash
uv run --group teaching jupyter lab
```

Elija el kernel **InvoiceOps ML Python 3.12** si Jupyter no lo selecciona
automáticamente.

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

## Dataset sintético

`invoiceops_ml.data.generate_synthetic_dataset()` crea el dataset versionado
`invoice-risk-v1` sin servicios externos. Una semilla fija produce los mismos
archivos CSV y metadatos, que registran la versión, el esquema de features, el
target, los tamaños de las particiones y la suma de verificación SHA-256 de cada
partición. La partición cronológica es 70% train, 15% validation y 15% test.

```python
from invoiceops_ml.data import generate_synthetic_dataset, seed_from_rut

dataset = generate_synthetic_dataset(
    seed=seed_from_rut("12345678-5"),
    rows=12_000,
)
```

El RUT debe estar previamente normalizado por InvoiceOps. Los datos generados se
escriben en el directorio ignorado `data/<dataset-version>/` y contienen solo
datos sintéticos.

## Notebook ML-04

Abra `notebooks/01_dataset.ipynb` con Jupyter y ejecute las celdas en orden. El
notebook genera o carga localmente `invoice-risk-v1` con una seed explícita,
revisa sus metadatos y muestra los splits cronológicos y la distribución del
target. Los datos sintéticos locales en `data/` están ignorados por Git.

## Notebook ML-05

Abra `notebooks/02_dummy.ipynb` después de configurar MLflow. El notebook genera
o carga `invoice-risk-v1`, entrena únicamente `DummyClassifier(strategy="prior")`
y registra parámetros y métricas de validation/test en MLflow. No registra un
model artifact ni aplica preprocessing: esos temas corresponden a los notebooks
de modelos entrenables posteriores.

Además de las variables `MLFLOW_*`, entregue el contexto académico no secreto
que proviene de InvoiceOps:

```bash
export INVOICEOPS_ORGANIZATION_SLUG=course-2027
export INVOICEOPS_OWNER_TYPE=user
export INVOICEOPS_OWNER_ID=ef14197c-8f5b-4aef-8fa7-310e4da998b7
export INVOICEOPS_CREATED_BY_RUT=12345678-5
```

Reemplace esos valores de ejemplo por el contexto vigente. No derive
`INVOICEOPS_OWNER_ID` desde un nombre o slug: debe ser el UUID de `User.id` para
trabajo individual o de `Group.id` para trabajo grupal. El notebook usa
`mlflow_config_from_env()` y `OwnershipContext`, por lo que funciona con la
configuración local o remota ya aprobada sin guardar URLs ni credenciales.

## Notebook ML-06

Abra `notebooks/03_logistic_regression.ipynb` después de configurar MLflow y el
mismo contexto académico no secreto de ML-05. El notebook carga los splits CSV
reproducibles existentes de `invoice-risk-v1`, entrena exclusivamente
`LogisticRegression` y mide accuracy, precision, recall y F1 en validation y
test.

El escalado de variables numéricas y la codificación de `country_risk` viven
dentro del `Pipeline` registrado. El pipeline se ajusta sólo con train, por lo
que validation y test no participan en el preprocessing aprendido y no hay
leakage. El run registra los parámetros, métricas, tags de ownership y el
artifact `model`, que contiene tanto preprocessing como clasificador.

## Notebook ML-07

Abra `notebooks/04_random_forest.ipynb` después de configurar MLflow y el mismo
contexto académico no secreto de ML-05. El notebook carga los splits CSV
reproducibles existentes de `invoice-risk-v1`, entrena exclusivamente
`RandomForestClassifier` y mide accuracy, precision, recall y F1 en validation y
test.

La codificación de `country_risk` vive dentro del `Pipeline` registrado junto con
el clasificador. El pipeline se ajusta sólo con train, por lo que validation y
test no participan en el preprocessing aprendido y no hay leakage. El run
registra los parámetros, métricas, tags de ownership y el artifact `model`, que
contiene tanto preprocessing como clasificador.

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
