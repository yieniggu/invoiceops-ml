from types import SimpleNamespace

import pytest

from invoiceops_ml import local_promotion
from invoiceops_ml.gate import GateResult
from invoiceops_ml.local_promotion import promote_local_run, select_and_promote_local

OWNER = {
    "INVOICEOPS_ORGANIZATION_SLUG": "app-07-local-demonstration",
    "INVOICEOPS_OWNER_TYPE": "user",
    "INVOICEOPS_OWNER_ID": "ef14197c-8f5b-4aef-8fa7-310e4da998b7",
    "INVOICEOPS_CREATED_BY_RUT": "111111111",
}
ENV = {**OWNER, "MLFLOW_TRACKING_URI": "http://127.0.0.1:5500"}


@pytest.fixture(autouse=True)
def isolate_mock_client_tracking(monkeypatch):
    """Fake-client tests must not redirect MLflow's registry for later notebook tests."""
    calls = []
    monkeypatch.setattr(local_promotion.mlflow, "set_tracking_uri", lambda uri: calls.append(("tracking", uri)))
    monkeypatch.setattr(local_promotion.mlflow, "set_registry_uri", lambda uri: calls.append(("registry", uri)))
    return calls


class Client:
    def __init__(self, *, tags=None, artifacts=None, versions=None):
        self.tags = (
            tags
            if tags is not None
            else {key.removeprefix("INVOICEOPS_").lower(): value for key, value in OWNER.items()}
        )
        self.artifacts = (
            [SimpleNamespace(path="model", is_dir=True)] if artifacts is None else artifacts
        )
        self.versions = [] if versions is None else versions
        self.actions = []

    def get_run(self, run_id):
        self.actions.append(("get_run", run_id))
        run = candidate(run_id, "hist-gradient-boosting")
        run.data.tags = {**self.tags, "mlflow.runName": "hist-gradient-boosting"}
        return run

    def get_experiment_by_name(self, name):
        self.actions.append(("experiment", name))
        return SimpleNamespace(experiment_id="owner-experiment")

    def list_artifacts(self, run_id):
        self.actions.append(("list_artifacts", run_id))
        return self.artifacts

    def search_model_versions(self, filter_string):
        self.actions.append(("search_model_versions", filter_string))
        return self.versions

    def set_registered_model_alias(self, name, alias, version):
        self.actions.append(("alias", name, alias, version))


def gate(run_id, client):
    assert run_id == "selected-run"
    return GateResult(run_id, "invoice-risk-gate-v1", {}, {}, {}, True)


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.org",
        "http://mlflow:5000",
        "http://0.0.0.0:5000",
        "file:///tmp/mlruns",
        "http://user:pass@127.0.0.1:5000",
        "http://127.0.0.1:5000/path",
    ],
)
def test_rejects_non_local_or_ambiguous_tracking_uri_before_client_use(uri):
    client = Client()
    with pytest.raises(ValueError, match="loopback"):
        promote_local_run("selected-run", {**ENV, "MLFLOW_TRACKING_URI": uri}, client, gate)
    assert client.actions == []


def test_rejects_missing_owner_and_workspace():
    client = Client()
    for environment in (
        {**ENV, "INVOICEOPS_OWNER_ID": ""},
        {**ENV, "MLFLOW_WORKSPACE": "other"},
        {**ENV, "MLFLOW_REGISTRY_URI": "https://production.example"},
    ):
        with pytest.raises(ValueError):
            promote_local_run("selected-run", environment, client, gate)
    assert client.actions == []


@pytest.mark.parametrize(
    "tags,artifacts",
    [
        ({"organization_slug": "other"}, None),
        (None, []),
        (None, [SimpleNamespace(path="model", is_dir=False)]),
    ],
)
def test_rejects_foreign_owner_or_missing_model_before_gate(tags, artifacts):
    client = Client(tags=tags, artifacts=artifacts)
    with pytest.raises(ValueError):
        promote_local_run("selected-run", ENV, client, gate)
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_gate_failure_cannot_register_or_assign_alias():
    client = Client()

    def failing_gate(run_id, client):
        return GateResult(run_id, "invoice-risk-gate-v1", {}, {}, {}, False)

    with pytest.raises(ValueError, match="FAIL"):
        promote_local_run("selected-run", ENV, client, failing_gate)
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_gate_result_for_a_different_run_cannot_assign_alias():
    client = Client()

    def mismatched_gate(run_id, client):
        return GateResult("another-run", "invoice-risk-gate-v1", {}, {}, {}, True)

    with pytest.raises(ValueError, match="mismatch"):
        promote_local_run("selected-run", ENV, client, mismatched_gate)
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


@pytest.mark.parametrize("name", ["logistic-regression", "random-forest", "rf-candidate-v1"])
def test_explicit_incompatible_gate_pass_cannot_promote(name):
    client = SelectionClient({"selected-run": candidate("selected-run", name, recall=0.99)})
    registered = []
    with pytest.raises(ValueError, match="compatible"):
        promote_local_run("selected-run", ENV, client, gate, lambda *args: registered.append(args))
    assert registered == []
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


@pytest.mark.parametrize("override", [
    {"experiment_id": "foreign"},
    {"status": "RUNNING"},
    {"tags": {"owner_id": "foreign"}},
    {"metrics": {"validation_recall": float("inf"), "validation_precision": 0.8,
                 "validation_f1": 0.8}},
])
def test_explicit_run_rejects_invalid_owner_status_or_metrics(override):
    run = candidate("selected-run", "hist-gradient-boosting", **override)
    client = SelectionClient({"selected-run": run})
    with pytest.raises(ValueError, match="eligible"):
        promote_local_run("selected-run", ENV, client, gate)
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_explicit_run_rejects_other_gate_version():
    client = Client()
    with pytest.raises(ValueError, match="FAIL"):
        promote_local_run("selected-run", ENV, client,
                          lambda run_id, _: GateResult(run_id, "other-gate", {}, {}, {}, True))
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_pass_reuses_only_same_run_version_and_assigns_canonical_alias():
    client = Client(
        versions=[
            SimpleNamespace(run_id="other-run", version="2"),
            SimpleNamespace(run_id="selected-run", version="3"),
        ]
    )
    version = promote_local_run("selected-run", ENV, client, gate)
    assert version == "3"
    assert client.actions[-1] == ("alias", "invoice-review-production", "champion", "3")


def test_pass_registers_exact_artifact_before_alias(isolate_mock_client_tracking):
    client = Client()
    registered = []

    def register(source, name):
        registered.append((source, name))
        return SimpleNamespace(version="4", run_id="selected-run")

    assert promote_local_run("selected-run", ENV, client, gate, register) == "4"
    assert registered == [("runs:/selected-run/model", "invoice-review-production")]
    assert isolate_mock_client_tracking == [
        ("tracking", ENV["MLFLOW_TRACKING_URI"]),
        ("registry", ENV["MLFLOW_TRACKING_URI"]),
    ]
    assert client.actions[-1] == ("alias", "invoice-review-production", "champion", "4")


def candidate(run_id, name, recall=0.8, precision=0.7, f1=0.75, *, tags=None, metrics=None,
              experiment_id="owner-experiment", status="FINISHED"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, experiment_id=experiment_id, status=status),
        data=SimpleNamespace(
            tags={**Client().tags, "mlflow.runName": name, **(tags or {})},
            metrics=metrics if metrics is not None else {
                "validation_recall": recall, "validation_precision": precision,
                "validation_f1": f1, "test_recall": 1.0,
            },
        ),
    )


class SelectionClient(Client):
    def __init__(self, runs, artifacts=None):
        super().__init__(artifacts=artifacts, versions=[SimpleNamespace(run_id=key, version="3") for key in runs])
        self.runs = runs

    def get_experiment_by_name(self, name):
        self.actions.append(("experiment", name))
        return SimpleNamespace(experiment_id="owner-experiment")

    def get_run(self, run_id):
        self.actions.append(("get_run", run_id))
        return self.runs[run_id]


def decisions(failed=()):
    calls = []

    def evaluate(run_id, client):
        calls.append(run_id)
        return GateResult(run_id, "invoice-risk-gate-v1", {},
                          {"validation_recall": client.runs[run_id].data.metrics["validation_recall"],
                           "validation_precision": client.runs[run_id].data.metrics["validation_precision"]},
                          {}, run_id not in failed)

    return evaluate, calls


def test_selection_ranks_pass_only_by_validation_and_rechecks_winner():
    runs = {
        "f1": candidate("f1", "random-forest", f1=0.85),
        "precision": candidate("precision", "logistic-regression", precision=0.8),
        "winner": candidate("winner", "hist-gradient-boosting", precision=0.8, f1=0.9),
        "recall": candidate("recall", "rf-candidate-v1", recall=0.9),
    }
    client = SelectionClient(runs)
    evaluate, calls = decisions(failed={"recall"})
    report = select_and_promote_local(list(reversed(runs)), ENV, client, evaluate)
    assert report == {"run_id": "winner", "model": "invoice-review-production",
                      "version": "3", "alias": "champion",
                      "gate_metrics": {"validation_recall": 0.8, "validation_precision": 0.8}}
    assert calls == ["winner", "winner"]
    assert client.actions[-1] == ("alias", "invoice-review-production", "champion", "3")


def test_selection_ties_resolve_by_run_id_not_input_order():
    client = SelectionClient({key: candidate(key, "hist-gradient-boosting") for key in ("z", "a")})
    evaluate, _ = decisions()
    assert select_and_promote_local(["z", "a"], ENV, client, evaluate)["run_id"] == "a"


@pytest.mark.parametrize("invalid", [
    candidate("bad", "dummy-baseline"),
    candidate("bad", "hist-gradient-boosting", tags={"owner_id": "another"}),
    candidate("bad", "hist-gradient-boosting", experiment_id="foreign"),
    candidate("bad", "hist-gradient-boosting", status="RUNNING"),
    candidate("bad", "hist-gradient-boosting", metrics={"validation_recall": 0.9, "validation_precision": 0.9}),
    candidate("bad", "hist-gradient-boosting", metrics={"validation_recall": float("nan"), "validation_precision": 0.9, "validation_f1": 0.9}),
])
def test_selection_excludes_invalid_candidates(invalid):
    client = SelectionClient({"bad": invalid, "good": candidate("good", "hist-gradient-boosting", recall=0.6)})
    evaluate, calls = decisions()
    assert select_and_promote_local(["bad", "good"], ENV, client, evaluate)["run_id"] == "good"
    assert calls == ["good", "good"]


def test_selection_excludes_missing_model_directory():
    client = SelectionClient({"bad": candidate("bad", "random-forest")}, artifacts=[])
    evaluate, calls = decisions()
    with pytest.raises(ValueError, match="No eligible Gate-PASS"):
        select_and_promote_local(["bad"], ENV, client, evaluate)
    assert calls == []
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_selection_no_pass_never_registers_or_assigns_alias():
    client = SelectionClient({"bad": candidate("bad", "rf-candidate-v1")})
    evaluate, _ = decisions(failed={"bad"})
    registered = []
    with pytest.raises(ValueError, match="No eligible Gate-PASS"):
        select_and_promote_local(["bad"], ENV, client, evaluate, lambda *args: registered.append(args))
    assert registered == []
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_selection_recheck_fail_does_not_mutate():
    client = SelectionClient({"good": candidate("good", "hist-gradient-boosting")})
    calls = []

    def changing_gate(run_id, client):
        calls.append(run_id)
        return GateResult(run_id, "invoice-risk-gate-v1", {}, {}, {}, len(calls) == 1)

    with pytest.raises(ValueError, match="FAIL"):
        select_and_promote_local(["good"], ENV, client, changing_gate)
    assert calls == ["good", "good"]
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_incompatible_gate_pass_cannot_outscore_or_replace_hgb():
    runs = {"rf": candidate("rf", "random-forest", recall=0.99),
            "lr": candidate("lr", "logistic-regression", recall=0.98),
            "hgb": candidate("hgb", "hist-gradient-boosting", recall=0.6)}
    client = SelectionClient(runs)
    evaluate, calls = decisions()
    assert select_and_promote_local(list(runs), ENV, client, evaluate)["run_id"] == "hgb"
    assert calls == ["hgb", "hgb"]


def test_without_hgb_pass_no_registration_or_alias_even_with_rf_pass():
    client = SelectionClient({"rf": candidate("rf", "random-forest"),
                              "hgb": candidate("hgb", "hist-gradient-boosting")})
    evaluate, calls = decisions(failed={"hgb"})
    registered = []
    with pytest.raises(ValueError, match="No eligible Gate-PASS"):
        select_and_promote_local(["rf", "hgb"], ENV, client, evaluate,
                                 lambda *args: registered.append(args))
    assert calls == ["hgb"]
    assert registered == []
    assert not any(action[0] in {"alias", "search_model_versions"} for action in client.actions)


def test_cli_selection_emits_only_safe_report(monkeypatch, capsys):
    import json

    received = []

    def select(run_ids):
        received.extend(run_ids)
        return {"run_id": "good", "model": "invoice-review-production", "version": "3",
                "alias": "champion", "gate_metrics": {"validation_recall": 0.8}}

    monkeypatch.setattr(local_promotion, "select_and_promote_local", select)
    assert local_promotion.main(["--candidate-run-id", "bad", "--candidate-run-id", "good"]) == 0
    assert received == ["bad", "good"]
    assert json.loads(capsys.readouterr().out) == select([])


def test_cli_explicit_run_remains_available(monkeypatch, capsys):
    import json

    monkeypatch.setattr(local_promotion, "run_quality_gate", lambda run_id, client:
                        GateResult(run_id, "invoice-risk-gate-v1", {}, {"validation_recall": 0.8}, {}, True))
    # Exercise the explicit CLI through its gate callback, without contacting MLflow.
    def promote(run_id, gate):
        gate(run_id, object())
        return "4"

    monkeypatch.setattr(local_promotion, "promote_local_run", promote)
    assert local_promotion.main(["--run-id", "selected-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "selected-run", "model": "invoice-review-production", "version": "4",
        "alias": "champion", "gate_metrics": {"validation_recall": 0.8},
    }
