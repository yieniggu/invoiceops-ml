import json
from pathlib import Path

import mlflow
import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

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


def test_hist_gradient_boosting_notebook_trains_and_logs_a_leakage_safe_pipeline() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "05_hist_gradient_boosting.ipynb"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in payload["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    assert "from sklearn.ensemble import HistGradientBoostingClassifier" in source
    assert "RandomForestClassifier" not in source
    assert "LogisticRegression" not in source
    assert "DummyClassifier" not in source
    assert "from sklearn.compose import ColumnTransformer" in source
    assert "from sklearn.pipeline import Pipeline" in source
    assert 'OneHotEncoder(handle_unknown="ignore", sparse_output=False)' in source
    assert "HistGradientBoostingClassifier(max_iter=200, random_state=202605)" in source
    assert "model.fit(train_features, train_target)" in source
    assert "model.fit(validation_features" not in source
    assert "model.fit(test_features" not in source
    assert 'classification_metrics("validation"' in source
    assert 'classification_metrics("test"' in source
    assert "configure_mlflow(mlflow_config_from_env())" in source
    assert "OwnershipContext" in source
    assert "set_run_ownership_tags(" in source
    assert 'with mlflow.start_run(run_name="hist-gradient-boosting")' in source
    assert "mlflow.log_params(model.get_params())" in source
    assert "mlflow.log_metrics(metrics)" in source
    assert "mlflow.sklearn.save_model(model, model_dir)" in source
    assert 'mlflow.log_artifacts(model_dir, artifact_path="model")' in source
    assert all(not cell.get("outputs") for cell in payload["cells"] if cell["cell_type"] == "code")


def test_hist_gradient_boosting_notebook_materializes_its_mlflow_run(
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
        repository / "notebooks" / "05_hist_gradient_boosting.ipynb", as_version=4
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


def test_model_comparison_notebook_recovers_runs_without_training() -> None:
    notebook = Path(__file__).parents[1] / "notebooks" / "06_model_comparison.ipynb"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in payload["cells"] if cell["cell_type"] == "code" for line in cell["source"]
    )

    assert "client.search_runs(" in source
    assert "MODEL_RUN_NAMES" in source
    assert "SUMMARY_METRICS" in source
    assert "DECISION_METRICS" in source
    assert "validation_recall" in source
    assert "validation_precision" in source
    assert "validation_f1" in source
    for tag in ("organization_slug", "owner_type", "owner_id", "created_by_rut"):
        assert tag in source
    assert "model.fit(" not in source
    assert "Classifier(" not in source
    assert all(not cell.get("outputs") for cell in payload["cells"] if cell["cell_type"] == "code")


def test_model_comparison_notebook_recovers_all_search_run_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    notebook = nbformat.read(repository / "notebooks" / "06_model_comparison.ipynb", as_version=4)
    notebook["cells"].insert(
        2,
        nbformat.v4.new_code_cell(
            """import mlflow
from types import SimpleNamespace


def run(run_id, run_name):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(
            tags={"mlflow.runName": run_name},
            metrics={
                "validation_accuracy": 0.75,
                "validation_precision": 0.70,
                "validation_recall": 0.80,
                "validation_f1": 0.75,
            },
        ),
    )


class Page(list):
    def __init__(self, runs, token):
        super().__init__(runs)
        self.token = token


class FakeClient:
    def __init__(self):
        self.calls = []

    def search_experiments(self):
        return [SimpleNamespace(experiment_id="comparison")]

    def search_runs(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["page_token"] is None:
            return Page(
                [run("first", "dummy-baseline"), run("second", "logistic-regression")],
                "next-page",
            )
        assert kwargs["page_token"] == "next-page"
        return Page(
            [run("third", "random-forest"), run("fourth", "hist-gradient-boosting")], None
        )


client = FakeClient()
mlflow.MlflowClient = lambda: client"""
        ),
    )
    notebook["cells"].append(nbformat.v4.new_code_cell("client.calls"))

    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=tmp_path)

    calls_output = notebook["cells"][-1]["outputs"][-1]["data"]["text/plain"]
    assert "'experiment_ids': ['comparison']" in calls_output
    assert "'run_view_type': 1" in calls_output
    assert "'page_token': None" in calls_output
    assert "'page_token': 'next-page'" in calls_output
    assert "filter_string" not in calls_output
    comparison_output = notebook["cells"][3]["outputs"][-1]["data"]["text/plain"]
    for run_name in (
        "dummy-baseline",
        "logistic-regression",
        "random-forest",
        "hist-gradient-boosting",
    ):
        assert f"'run_name': '{run_name}'" in comparison_output


def test_model_comparison_notebook_ranks_existing_validation_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("comparison")

    ownership_tags = {
        "organization_slug": "course-2027",
        "owner_type": "group",
        "owner_id": "3515a7c6-baa4-44aa-a433-e7c52d79a57d",
        "created_by_rut": "12.345.678-5",
    }
    for run_name, recall, precision, f1 in (
        ("dummy-baseline", 0.40, 0.75, 0.50),
        ("logistic-regression", 0.80, 0.65, 0.70),
        ("random-forest", 0.80, 0.72, 0.75),
        ("hist-gradient-boosting", 0.78, 0.80, 0.79),
    ):
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.set_tags(ownership_tags)
            mlflow.log_metrics(
                {
                    "validation_precision": precision,
                    "validation_recall": recall,
                    "validation_f1": f1,
                }
            )

    notebook = nbformat.read(
        repository / "notebooks" / "06_model_comparison.ipynb", as_version=4
    )
    NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=tmp_path)

    comparison_output = notebook["cells"][2]["outputs"][-1]["data"]["text/plain"]
    for tag, value in ownership_tags.items():
        assert f"'{tag}': '{value}'" in comparison_output
    candidate_output = notebook["cells"][4]["outputs"][-1]["data"]["text/plain"]
    assert "'run_name': 'random-forest'" in candidate_output


def test_model_comparison_notebook_requires_all_model_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[1]
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("comparison")

    for run_name in ("dummy-baseline", "logistic-regression", "random-forest"):
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_metrics(
                {
                    "validation_precision": 0.75,
                    "validation_recall": 0.80,
                    "validation_f1": 0.77,
                }
            )

    notebook = nbformat.read(
        repository / "notebooks" / "06_model_comparison.ipynb", as_version=4
    )
    with pytest.raises(CellExecutionError, match="Missing required model runs: hist-gradient-boosting"):
        NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=tmp_path)
