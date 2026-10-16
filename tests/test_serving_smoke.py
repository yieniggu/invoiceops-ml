"""ML-15 pre-implementation contract; no live service or production substitute.

Proposed local API (not a change to the provider API):
invoiceops_ml.serving.run_serving_smoke() reads MODEL_API_URL and returns a dict
with ``health``, ``prediction`` (each response's own metadata), and ``warnings``
(a list of safe strings, nonempty when version/run changes between requests).
ServingSmokeError exposes ``category``: configuration, transport, http, schema,
or response_too_large. Its rendered traceback and captured diagnostics are safe.
The notebook delegates to this API and displays both responses and any warning.

The approved stdlib transport seam is HTTP(S)Handler's open method: real urllib
opener policy, orchestration and validation remain under test. Response doubles
only supply HTTP bytes/status/headers. No module import occurs at collection.
Bootstrap RED demonstrates absent artifacts, NOT working runtime validators.
"""

import ast
import importlib
import io
import json
import socket
import subprocess
import traceback
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.response import addinfourl

import pytest

ROOT = Path(__file__).parents[1]
BASE_URL = "https://model.invalid"
SECRET = "sensitive-sentinel-ml15"
LIMIT = 64 * 1024
FEATURES = {
    "invoice_amount_cents": 125000,
    "vendor_tenure_days": 365,
    "previous_incidents_12m": 0,
    "amount_vs_vendor_median": 1.2,
    "has_purchase_order": True,
    "three_way_match": True,
    "bank_account_recently_changed": False,
    "country_risk": "low",
}
HEALTH = {
    "model_id": "invoice-review",
    "registered_model": "invoice-review-production",
    "alias": "champion",
    "model_version": "7",
    "run_id": "run-health",
}
PREDICTION = {
    "model_id": "invoice-review",
    "model_version": "7",
    "run_id": "run-health",
    "probability": 0.25,
}


class WireResponse(addinfourl):
    """HTTP response bytes with bounded-read evidence, not serving behavior."""

    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        message = Message()
        for name, value in (headers or {}).items():
            message[name] = value
        super().__init__(io.BytesIO(body), message, BASE_URL, status)
        self.msg = "HTTP response"
        self.read_sizes = []

    def read(self, size=-1):
        assert 0 < size <= LIMIT + 1, "Response reads must be explicitly bounded"
        self.read_sizes.append(size)
        return self.fp.read(size)


def response(payload, status=200, headers=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return WireResponse(body, status, headers)


class Wire:
    def __init__(self):
        self.replies = []
        self.requests = []

    def open(self, handler, request):
        self.requests.append(request)
        assert request.timeout == 10, "Use a 10s socket timeout, not a total deadline"
        assert not request.has_proxy(), "Do not inherit environment proxies"
        assert len(self.requests) <= 2, "No retries, redirects or administrative requests"
        assert self.replies, "Unexpected HTTP request"
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def wire(monkeypatch):
    transport = Wire()

    def open_response(handler, request):
        return transport.open(handler, request)

    def forbidden(*args, **kwargs):
        pytest.fail("No real sockets or child processes are permitted")

    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", open_response)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", open_response)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setenv("MODEL_API_URL", BASE_URL)
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, f"http://{SECRET}.invalid:8888")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    return transport


def serving():
    return importlib.import_module("invoiceops_ml.serving")


def assert_safe_error(module, category, capsys, caplog):
    with pytest.raises(module.ServingSmokeError) as caught:
        module.run_serving_smoke()
    assert caught.value.category == category
    output = capsys.readouterr()
    diagnostics = "".join(traceback.format_exception(caught.value))
    diagnostics += output.out + output.err + caplog.text
    assert SECRET not in diagnostics
    assert BASE_URL not in diagnostics


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "   ",
        "model.invalid",
        "file:///tmp/model",
        f"https://user:{SECRET}@model.invalid",
        f"https://model.invalid/?token={SECRET}",
        f"https://model.invalid/#{SECRET}",
        "https://model.invalid:invalid",
    ],
    ids=[
        "absent",
        "empty",
        "blank",
        "relative",
        "scheme",
        "credentials",
        "query",
        "fragment",
        "port",
    ],
)
def test_configuration_is_explicit_and_rejected_before_io(url, wire, monkeypatch, capsys, caplog):
    module = serving()
    if url is None:
        monkeypatch.delenv("MODEL_API_URL")
    else:
        monkeypatch.setenv("MODEL_API_URL", url)
    assert_safe_error(module, "configuration", capsys, caplog)
    assert wire.requests == []


@pytest.mark.parametrize("probability", [0, 0.25, 1], ids=["zero", "fraction", "one"])
def test_health_then_one_predict_with_exact_synthetic_features(probability, wire):
    module = serving()
    prediction = {**PREDICTION, "probability": probability}
    wire.replies = [response(HEALTH), response(prediction)]
    result = module.run_serving_smoke()
    assert result["health"] == HEALTH
    assert result["prediction"] == prediction
    assert result["warnings"] == []
    assert [(r.get_method(), r.full_url) for r in wire.requests] == [
        ("GET", BASE_URL + "/models/invoice-review/health"),
        ("POST", BASE_URL + "/models/invoice-review/predict"),
    ]
    assert wire.requests[0].data is None
    assert json.loads(wire.requests[1].data) == FEATURES
    assert wire.requests[1].get_header("Content-type") == "application/json"
    for request in wire.requests:
        assert not any(
            name.lower()
            in {"authorization", "proxy-authorization", "x-model-admin-token", "cookie"}
            for name, _ in request.header_items()
        )
    assert wire.replies == []


def test_refresh_between_calls_preserves_both_snapshots_and_warns(wire):
    module = serving()
    changed = {**PREDICTION, "model_version": "8", "run_id": "run-predict"}
    wire.replies = [response(HEALTH), response(changed)]
    result = module.run_serving_smoke()
    assert result["health"] == HEALTH
    assert result["prediction"] == changed
    assert result["warnings"] and all(isinstance(item, str) for item in result["warnings"])
    assert len(wire.requests) == 2


def test_response_limit_accepts_exactly_64_kib_of_valid_json(wire):
    module = serving()
    body = json.dumps(HEALTH).encode()
    wire.replies = [response(body + b" " * (LIMIT - len(body))), response(PREDICTION)]
    assert module.run_serving_smoke()["prediction"] == PREDICTION
    assert len(wire.requests) == 2


@pytest.mark.parametrize("stage", ["health", "predict"])
@pytest.mark.parametrize(
    "failure,category",
    [
        (TimeoutError(SECRET), "transport"),
        (OSError(SECRET), "transport"),
        ("unauthorized", "http"),
        ("unavailable", "http"),
        ("redirect", "http"),
        ("invalid-json", "schema"),
        ("invalid-utf8", "schema"),
        ("non-object", "schema"),
        ("oversized", "response_too_large"),
    ],
    ids=["timeout", "connection", "401", "503", "302", "json", "utf8", "array", "size"],
)
def test_failures_are_categorical_safe_and_never_retried(
    stage, failure, category, wire, capsys, caplog
):
    module = serving()
    if isinstance(failure, Exception):
        reply = failure
    else:
        reply = {
            "unauthorized": lambda: response({"detail": SECRET}, 401, {"X-Secret": SECRET}),
            "unavailable": lambda: response({"detail": SECRET}, 503),
            "redirect": lambda: response(b"", 302, {"Location": f"https://{SECRET}.invalid"}),
            "invalid-json": lambda: response(SECRET.encode()),
            "invalid-utf8": lambda: response(b"\xff"),
            "non-object": lambda: response([SECRET]),
            "oversized": lambda: response(b" " * (LIMIT + 1)),
        }[failure]()
    wire.replies = ([response(HEALTH)] if stage == "predict" else []) + [reply]
    assert_safe_error(module, category, capsys, caplog)
    assert len(wire.requests) == (2 if stage == "predict" else 1)
    assert wire.replies == []


@pytest.mark.parametrize("stage", ["health", "predict"])
@pytest.mark.parametrize(
    "field,value",
    [("model_id", "other-model"), ("model_version", ""), ("run_id", None)],
)
def test_each_response_requires_its_own_valid_metadata(stage, field, value, wire, capsys, caplog):
    module = serving()
    payload = dict(HEALTH if stage == "health" else PREDICTION)
    payload[field] = value
    wire.replies = ([response(HEALTH)] if stage == "predict" else []) + [response(payload)]
    assert_safe_error(module, "schema", capsys, caplog)
    assert len(wire.requests) == (2 if stage == "predict" else 1)


@pytest.mark.parametrize(
    "probability",
    [True, "0.5", None, -0.01, 1.01, float("nan"), float("inf")],
    ids=["bool", "string", "null", "negative", "above-one", "nan", "infinity"],
)
def test_prediction_probability_must_be_finite_numeric_and_in_range(
    probability, wire, capsys, caplog
):
    module = serving()
    wire.replies = [response(HEALTH), response({**PREDICTION, "probability": probability})]
    assert_safe_error(module, "schema", capsys, caplog)
    assert len(wire.requests) == 2


def test_notebook_is_clean_spanish_and_delegates_without_operational_side_effects(wire, capsys):
    notebook = ROOT / "notebooks" / "08_serving_smoke.ipynb"
    assert notebook.is_file(), "ML-15 notebook is not implemented"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    nbformat.validate(nbformat.from_dict(payload))
    assert payload["nbformat"] == 4
    markdown = "\n".join(
        "".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "markdown"
    ).lower()
    assert "modelo" in markdown and "predicci" in markdown
    cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert cells
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in cells)
    source = "\n".join("".join(cell["source"]) for cell in cells)
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "run_serving_smoke"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_serving_smoke"
        )
        for node in ast.walk(tree)
    )
    # Static tripwires complement (not replace) the exact two-request wire assertion.
    forbidden = {
        "mlflow",
        "subprocess",
        "uvicorn",
        "train",
        "fit",
        "refresh",
        "promote",
        "load_model",
        "download_artifacts",
    }
    for code in (source, (ROOT / "src" / "invoiceops_ml" / "serving.py").read_text()):
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & forbidden
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or "").split(".")) & forbidden
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", "")
                )
                assert name not in forbidden
    changed = {**PREDICTION, "model_version": "8", "run_id": "run-predict"}
    wire.replies = [response(HEALTH), response(changed)]
    shell = InteractiveShell(user_ns={"__name__": "__main__"})
    for cell in cells:
        execution = shell.run_cell("".join(cell["source"]), store_history=False)
        assert execution.success, "Notebook cells must execute in order without errors"
    output = capsys.readouterr().out
    for value in ("invoice-review", "7", "8", "run-health", "run-predict"):
        assert value in output
    assert any(word in output.lower() for word in ("advertencia", "cambi", "warning"))
    assert len(wire.requests) == 2
