import json
from pathlib import Path

import mlflow
import nbformat
import pytest
from nbclient import NotebookClient

from invoiceops_ml.data import generate_synthetic_dataset


def test_dummy_notebook_trains_only_the_baseline_and_logs_its_run() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "02_dummy.ipynb"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in payload["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    assert "from sklearn.dummy import DummyClassifier" in source
    assert "DummyClassifier(strategy=\"prior\")" in source
    assert "configure_mlflow(mlflow_config_from_env())" in source
    assert "set_run_ownership_tags(" in source
    assert "with mlflow.start_run(run_name=\"dummy-baseline\")" in source
    assert "mlflow.log_params(model.get_params())" in source
    assert "classification_metrics('validation'" in source
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert f"f'{{prefix}}_{metric}'" in source


def test_logistic_regression_notebook_trains_and_logs_a_leakage_safe_pipeline() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "03_logistic_regression.ipynb"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in payload["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    assert "from sklearn.linear_model import LogisticRegression" in source
    assert "from sklearn.compose import ColumnTransformer" in source
    assert "from sklearn.pipeline import Pipeline" in source
    assert "StandardScaler()" in source
    assert 'OneHotEncoder(handle_unknown="ignore")' in source
    assert "LogisticRegression(max_iter=1_000, random_state=202605)" in source
    assert "model.fit(train_features, train_target)" in source
    assert "model.fit(validation_features" not in source
    assert "model.fit(test_features" not in source
    assert 'classification_metrics("validation"' in source
    assert 'classification_metrics("test"' in source
    assert "configure_mlflow(mlflow_config_from_env())" in source
    assert "set_run_ownership_tags(" in source
    assert 'with mlflow.start_run(run_name="logistic-regression")' in source
    assert "mlflow.log_params(model.get_params())" in source
    assert "mlflow.log_metrics(metrics)" in source
    assert "mlflow.sklearn.save_model(model, model_dir)" in source
    assert 'mlflow.log_artifacts(model_dir, artifact_path="model")' in source
    assert all(not cell.get("outputs") for cell in payload["cells"] if cell["cell_type"] == "code")


def test_logistic_regression_notebook_materializes_its_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    dataset = generate_synthetic_dataset(seed=202605, rows=100, output_root=tmp_path / "data")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    ownership_tags = {
        "organization_slug": "course-2027",
        "owner_type": "user",
        "owner_id": "ef14197c-8f5b-4aef-8fa7-310e4da998b7",
        "created_by_rut": "12.345.678-5",
    }
    monkeypatch.setenv("INVOICEOPS_DATASET_DIR", str(dataset))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    for name, value in ownership_tags.items():
        monkeypatch.setenv(f"INVOICEOPS_{name.upper()}", value)

    notebook = nbformat.read(
        repository / "notebooks" / "03_logistic_regression.ipynb", as_version=4
    )
    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=tmp_path)

    experiment = mlflow.MlflowClient(tracking_uri=tracking_uri).get_experiment_by_name("Default")
    assert experiment is not None
    runs = mlflow.MlflowClient(tracking_uri=tracking_uri).search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert set(run.data.metrics) >= {
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    }
    assert {name: run.data.tags[name] for name in ownership_tags} == ownership_tags
    artifacts = mlflow.MlflowClient(tracking_uri=tracking_uri).list_artifacts(run.info.run_id)
    assert [artifact.path for artifact in artifacts] == ["model"]


def test_random_forest_notebook_trains_and_logs_a_leakage_safe_pipeline() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "04_random_forest.ipynb"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in payload["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    assert "from sklearn.ensemble import RandomForestClassifier" in source
    assert "LogisticRegression" not in source
    assert "DummyClassifier" not in source
    assert "from sklearn.compose import ColumnTransformer" in source
    assert "from sklearn.pipeline import Pipeline" in source
    assert 'OneHotEncoder(handle_unknown="ignore")' in source
    assert "RandomForestClassifier(n_estimators=200, random_state=202605, n_jobs=-1)" in source
    assert "model.fit(train_features, train_target)" in source
    assert "model.fit(validation_features" not in source
    assert "model.fit(test_features" not in source
    assert 'classification_metrics("validation"' in source
    assert 'classification_metrics("test"' in source
    assert "configure_mlflow(mlflow_config_from_env())" in source
    assert "OwnershipContext" in source
    assert "set_run_ownership_tags(" in source
    assert 'with mlflow.start_run(run_name="random-forest")' in source
    assert "mlflow.log_params(model.get_params())" in source
    assert "mlflow.log_metrics(metrics)" in source
    assert "mlflow.sklearn.save_model(model, model_dir)" in source
    assert 'mlflow.log_artifacts(model_dir, artifact_path="model")' in source
    assert all(not cell.get("outputs") for cell in payload["cells"] if cell["cell_type"] == "code")


def test_random_forest_notebook_materializes_its_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    dataset = generate_synthetic_dataset(seed=202605, rows=100, output_root=tmp_path / "data")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    ownership_tags = {
        "organization_slug": "course-2027",
        "owner_type": "user",
        "owner_id": "ef14197c-8f5b-4aef-8fa7-310e4da998b7",
        "created_by_rut": "12.345.678-5",
    }
    monkeypatch.setenv("INVOICEOPS_DATASET_DIR", str(dataset))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    for name, value in ownership_tags.items():
        monkeypatch.setenv(f"INVOICEOPS_{name.upper()}", value)

    notebook = nbformat.read(
        repository / "notebooks" / "04_random_forest.ipynb", as_version=4
    )
    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=tmp_path)

    experiment = mlflow.MlflowClient(tracking_uri=tracking_uri).get_experiment_by_name("Default")
    assert experiment is not None
    runs = mlflow.MlflowClient(tracking_uri=tracking_uri).search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert set(run.data.metrics) >= {
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    }
    assert {name: run.data.tags[name] for name in ownership_tags} == ownership_tags
    artifacts = mlflow.MlflowClient(tracking_uri=tracking_uri).list_artifacts(run.info.run_id)
    assert [artifact.path for artifact in artifacts] == ["model"]
