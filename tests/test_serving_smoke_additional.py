"""Supplementary ML-15 evidence; the original immutable contract gate stays separate.

The original WireResponse calls a nonexistent super().read(). This independent
wire response delegates to its BytesIO explicitly; it does not patch that gate.
"""

import http.client
import json
import traceback
from urllib.error import HTTPError

import pytest
import test_serving_smoke as contract

from invoiceops_ml.serving import ServingSmokeError, run_serving_smoke

wire = contract.wire


class Response(contract.WireResponse):
    def read(self, size=-1):
        assert 0 < size <= contract.LIMIT + 1
        self.read_sizes.append(size)
        return self.fp.read(min(size, 1024))


def response(payload, status=200, headers=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return Response(body, status, headers)


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1]suffix",
        "https://bad_host",
        "https://model.invalid:",
        "https://model.invalid:0",
        "https://model.invalid:65536",
        "https://model.invalid\n",
        "https://model.invalid\\path",
        "https://model.invalid?",
        "https://model.invalid#",
        "https://model.invalid/../admin",
        "https://model.invalid/%0a",
    ],
)
def test_additional_invalid_configuration_precedes_io(url, wire, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", url)
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "configuration"
    assert not wire.requests


@pytest.mark.parametrize("base", ["http://127.0.0.1:8000", "http://[::1]:8000", contract.BASE_URL])
@pytest.mark.parametrize("suffix", ["", "/", "/gateway///"])
def test_exact_requests_base_paths_metadata_and_closed_responses(base, suffix, wire, monkeypatch):
    monkeypatch.setenv("MODEL_API_URL", base + suffix)
    health, prediction = response(contract.HEALTH), response(contract.PREDICTION)
    wire.replies = [health, prediction]
    assert run_serving_smoke() == {
        "health": contract.HEALTH,
        "prediction": contract.PREDICTION,
        "warnings": [],
    }
    assert [(item.get_method(), item.full_url) for item in wire.requests] == [
        ("GET", (base + suffix).rstrip("/") + "/models/invoice-review/health"),
        ("POST", (base + suffix).rstrip("/") + "/models/invoice-review/predict"),
    ]
    assert json.loads(wire.requests[1].data) == contract.FEATURES
    assert health.closed and prediction.closed
    for request in wire.requests:
        assert not any(
            key.lower() in {"authorization", "proxy-authorization", "cookie", "x-model-admin-token"}
            for key, _ in request.header_items()
        )


@pytest.mark.parametrize("stage", ["health", "predict"])
@pytest.mark.parametrize(
    "failure,category",
    [
        ("timeout", "transport"),
        ("protocol", "transport"),
        ("401", "http"),
        ("503", "http"),
        ("redirect", "http"),
        ("json", "schema"),
        ("utf8", "schema"),
        ("array", "schema"),
        ("oversized", "response_too_large"),
    ],
)
def test_categorical_safe_failures_close_responses(stage, failure, category, wire, capsys, caplog):
    reply = {
        "timeout": lambda: TimeoutError(contract.SECRET),
        "protocol": lambda: http.client.IncompleteRead(contract.SECRET.encode()),
        "401": lambda: response({"detail": contract.SECRET}, 401),
        "503": lambda: response({"detail": contract.SECRET}, 503),
        "redirect": lambda: response(b"", 302, {"Location": f"https://{contract.SECRET}.invalid"}),
        "json": lambda: response(contract.SECRET.encode()),
        "utf8": lambda: response(b"\xff"),
        "array": lambda: response([contract.SECRET]),
        "oversized": lambda: response(b" " * (contract.LIMIT + 1)),
    }[failure]()
    replies = ([response(contract.HEALTH)] if stage == "predict" else []) + [reply]
    wire.replies = replies.copy()
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == category
    diagnostics = "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    diagnostics += output.out + output.err + caplog.text
    assert contract.SECRET not in diagnostics and contract.BASE_URL not in diagnostics
    assert len(wire.requests) == len(replies)
    assert all(item.closed for item in replies if isinstance(item, Response))


def test_direct_http_error_is_closed_without_reading(wire):
    body = response({"detail": contract.SECRET})
    wire.replies = [HTTPError(contract.BASE_URL, 401, contract.SECRET, {}, body)]
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "http"
    assert body.closed and not body.read_sizes
    assert contract.SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_location_is_never_parsed(status, wire):
    reply = response(b"", status, {"Location": "https://[" + contract.SECRET})
    wire.replies = [reply]
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "http"
    assert reply.closed and not reply.read_sizes
    assert len(wire.requests) == 1
    assert contract.SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "field", ["model_id", "model_version", "run_id", "registered_model", "alias"]
)
def test_invalid_health_metadata_stops_prediction(field, wire):
    health = response({**contract.HEALTH, field: "  "})
    wire.replies = [health]
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "schema"
    assert len(wire.requests) == 1 and health.closed


@pytest.mark.parametrize("field", ["model_id", "model_version", "run_id"])
def test_invalid_prediction_metadata_is_rejected(field, wire):
    wire.replies = [response(contract.HEALTH), response({**contract.PREDICTION, field: None})]
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "schema"


@pytest.mark.parametrize("value", [True, "0.5", None, -1, 2, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(value, wire):
    wire.replies = [
        response(contract.HEALTH),
        response({**contract.PREDICTION, "probability": value}),
    ]
    with pytest.raises(ServingSmokeError) as caught:
        run_serving_smoke()
    assert caught.value.category == "schema"


@pytest.mark.parametrize("value", [0, 0.25, 1])
def test_valid_probability_with_inclusive_limit_and_changed_snapshot(value, wire):
    health_body = json.dumps(contract.HEALTH).encode()
    health = response(health_body + b" " * (contract.LIMIT - len(health_body)))
    prediction = {**contract.PREDICTION, "probability": value, "run_id": "run-next"}
    wire.replies = [health, response(prediction)]
    result = run_serving_smoke()
    assert result["health"] == contract.HEALTH
    assert result["prediction"] == prediction
    assert result["warnings"]
    assert len(health.read_sizes) > 1 and health.closed


def test_notebook_cells_execute_with_independent_transport(wire, capsys):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell

    notebook = nbformat.read(contract.ROOT / "notebooks/08_serving_smoke.ipynb", as_version=4)
    nbformat.validate(notebook)
    wire.replies = [
        response(contract.HEALTH),
        response({**contract.PREDICTION, "model_version": "8", "run_id": "run-next"}),
    ]
    shell = InteractiveShell(user_ns={"__name__": "__main__"})
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            assert shell.run_cell(cell.source, store_history=False).success
    output = capsys.readouterr().out
    for expected in (
        "invoice-review",
        "7",
        "8",
        "run-health",
        "run-next",
        "Probabilidad",
        "Advertencia",
    ):
        assert expected in output
    assert len(wire.requests) == 2
