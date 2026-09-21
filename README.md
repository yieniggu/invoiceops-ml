# InvoiceOps ML

Workspace principal de investigación para los alumnos de InvoiceOps. Los
notebooks explican el flujo didáctico y el código reusable vive en `src/`.

## Alcance actual

Este repositorio incluye la configuración reusable de MLflow de ML-01, los tags
de ownership de ML-02, el generador de datasets de ML-03 y los notebooks
didácticos de ML-04 a ML-09. No incluye Docker ni infraestructura de MLflow.

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
`../dev/tickets/INT-02_ownership_academico_mlflow.md`.

## Convenciones de recursos MLflow

Los recursos de cada owner no colisionan dentro de su Workspace porque el nombre
canónico incorpora su identidad estable:

| Tipo de trabajo | Experiment | Registered Model |
| --- | --- | --- |
| Individual | `student/<created_by_rut>/invoice-risk` | `student-<created_by_rut>-invoice-review` |
| Grupal | `group/<Group.id>/invoice-risk` | `group-<Group.id>-invoice-review` |

Para trabajo individual, `created_by_rut` es siempre la identidad del owner del
recurso: no hay delegación. El `owner_id` UUID se conserva en los tags para
resolver el usuario de InvoiceOps, pero no participa en el nombre individual.
Para trabajo grupal, `Group.id` UUID es la única identidad de nombres y roles;
nunca use `Group.name` ni un slug.

Los notebooks entrenables seleccionan el Experiment correcto antes de abrir el
run. Al registrar un modelo owner-scoped, reutilice los helpers para no repetir
ni derivar convenciones:

```python
import mlflow

from invoiceops_ml.ownership import (
    owner_registered_model_name,
    set_registered_model_ownership_tags,
)

client = mlflow.MlflowClient()
model_name = owner_registered_model_name(ownership_context)
client.create_registered_model(model_name)
set_registered_model_ownership_tags(model_name, ownership_context, client)
```

`invoice-review-production` es el único Registered Model compartido. No tiene
un owner académico y no debe recibir los tags de ownership ni un `owner_type`
inventado como `shared`: el contrato sólo admite `user` y `group`. Su promoción
y automatización pertenecen a tickets posteriores.

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

## Notebook ML-08

Abra `notebooks/05_hist_gradient_boosting.ipynb` después de configurar MLflow y
el mismo contexto académico no secreto de ML-05. El notebook carga los splits
CSV reproducibles existentes de `invoice-risk-v1`, entrena exclusivamente
`HistGradientBoostingClassifier` y mide accuracy, precision, recall y F1 en
validation y test.

La codificación de `country_risk` vive dentro del `Pipeline` registrado y se
materializa como matriz densa, requisito de `HistGradientBoostingClassifier`.
El pipeline se ajusta sólo con train, por lo que validation y test no participan
en el preprocessing aprendido y no hay leakage. El run registra los parámetros,
métricas, tags de ownership y el artifact `model`, que contiene tanto
preprocessing como clasificador.

## Notebook ML-09

Abra `notebooks/06_model_comparison.ipynb` después de registrar runs con los
notebooks ML-05 a ML-08. El notebook recupera esos runs existentes y los ordena
por `validation_recall`, `validation_precision` y `validation_f1`; no genera
datos ni reentrena modelos.

Priorice **Compare Runs** en la UI de MLflow: filtre los cuatro `run_name`,
selecciónelos y compare métricas, parámetros y tags de ownership. El notebook
no recomienda un candidate si falta alguno de los cuatro runs, e informa los
`run_name` ausentes. El candidate sugerido requiere y ordena sólo
`validation_recall`, `validation_precision` y `validation_f1`;
`validation_accuracy` permanece como dato informativo y no influye en la
selección. El alumno debe registrar el `run_id` y justificar el trade-off
observado antes de continuar con el Quality Gate de ML-10. ML-09 no ejecuta
Gates ni promueve modelos.

## ML-10 quality gate

`invoiceops_ml/invoice-risk-gate-v1.json` is the versioned baseline configuration shipped with the package. It
evaluates only `validation_recall >= 0.18` and `validation_precision >= 0.48`
for one explicit MLflow run. Run it from the repository root after configuring
MLflow:

```bash
uv run invoiceops-quality-gate --run-id <mlflow-run-id>
```

The command emits JSON with the run ID, gate version, thresholds, observed
metrics, individual checks, and the overall `passed` decision. It exits `0` for
PASS and `1` for FAIL, so a later workflow can invoke it without changing its
evaluation logic. A different versioned JSON file can be selected with
`--config path/to/gate.json`.

The reusable Python interface is `invoiceops_ml.gate.run_quality_gate()`. The
gate reads a run and reports eligibility only. It does not create model versions,
set aliases, call the Model Registry, or promote any model.

## ML-11 candidate training

`config/candidates/rf-candidate-v1.json` is the versioned, reviewed contract for
the Random Forest candidate. It declares the dataset generator inputs, feature
schema, target, model, and exact parameters. It intentionally does not contain
precomputed split hashes: the CLI materializes the dataset from its declared
`seed` and `rows`, then verifies both the metadata lineage and the SHA-256 hash
of each split before training.

Run it from the repository root after configuring MLflow and the same non-secret
ownership context used by the training notebooks:

```bash
uv run invoiceops-train-candidate --spec config/candidates/rf-candidate-v1.json
```

The default materialization directory is `notebooks/data/invoice-risk-v1`. Use
`--dataset-dir` to select another directory ending in `invoice-risk-v1`:

```bash
uv run invoiceops-train-candidate \
  --spec config/candidates/rf-candidate-v1.json \
  --dataset-dir /tmp/invoice-risk-v1
```

The command fits its preprocessing pipeline and `RandomForestClassifier` only on
the train split, evaluates validation and test splits, and writes a new MLflow
run. The run includes ownership tags, candidate and dataset-lineage tags,
parameters, metrics, the candidate JSON under `candidate_specification`, and the
fitted pipeline under `model`. It does not run a quality gate, register or
promote a model, interact with the Model Registry, or invoke notebooks.

## Notebook ML-13

Abra `notebooks/07_registry_gate_and_promotion.ipynb` después de seleccionar un
run existente con un artifact `model` y tags de ownership que coincidan con el
contexto académico vigente. Entregue su ID sin escribirlo en el notebook:

```bash
export INVOICEOPS_SELECTED_RUN_ID=<mlflow-run-id>
```

El notebook deriva el nombre canónico owner-scoped del Registered Model, valida el
ownership del run antes de registrar `runs:/<run-id>/model` como una Model Version y
escribe los tags estables de ownership. Si ya existe una versión de ese mismo run
para el mismo Registered Model, la reutiliza. Luego ejecuta el Quality Gate
versionado existente para ese run. Muestra el ID del run, el nombre del Registered
Model, la Model Version y el reporte serializable del Gate.

En la UI de MLflow, asigne manualmente `challenger` a la versión registrada bajo
revisión. Un PASS del Gate sólo acredita elegibilidad: no registra una versión, no
asigna un alias, no aprueba una decisión de revisor ni promueve `champion`. Asigne
`champion` en la UI sólo después de registrar una decisión explícita de un revisor
autorizado.

## Estructura

```text
notebooks/          Material didáctico
src/invoiceops_ml/  Código Python reusable
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
