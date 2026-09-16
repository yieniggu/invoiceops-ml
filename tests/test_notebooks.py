import json
from pathlib import Path


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
