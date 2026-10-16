"""Bounded, unauthenticated smoke check of an explicitly configured Model API."""

import http.client
import ipaddress
import json
import math
import os
import re
from typing import Literal
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, ProxyHandler, Request, build_opener

MAX_RESPONSE_BYTES = 64 * 1024
SOCKET_TIMEOUT_SECONDS = 10
ErrorCategory = Literal["configuration", "transport", "http", "schema", "response_too_large"]


class ServingSmokeError(Exception):
    """Safe failure with category configuration, transport, http, schema, or response_too_large."""

    def __init__(self, category: ErrorCategory) -> None:
        self.category = category
        super().__init__(f"Serving smoke failed ({category})")


class _NoRedirects(HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        # Reject before urllib parses an untrusted Location or reads the body.
        raise HTTPError(req.full_url, code, "Redirect refused", headers, fp)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def _base_url() -> str:
    value = os.environ.get("MODEL_API_URL", "")
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ServingSmokeError("configuration")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme not in ("http", "https")
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or "?" in value
            or "#" in value
            or "\\" in value
            or parsed.netloc.endswith(":")
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ServingSmokeError("configuration")
        if ":" in host:
            ipaddress.IPv6Address(host)
            if "%" in host or not re.fullmatch(r"\[[0-9A-Fa-f:.]+\](?::[0-9]+)?", parsed.netloc):
                raise ServingSmokeError("configuration")
        elif (
            not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in host.removesuffix(".").split(".")
            )
            or len(host) > 253
        ):
            raise ServingSmokeError("configuration")
        if not re.fullmatch(r"[A-Za-z0-9/_~.\-]*", parsed.path) or any(
            part in (".", "..") for part in parsed.path.split("/")
        ):
            raise ServingSmokeError("configuration")
    except ValueError:
        raise ServingSmokeError("configuration") from None
    return value.rstrip("/")


def _request_json(opener: OpenerDirector, request: Request) -> dict[str, object]:
    try:
        try:
            response = opener.open(request, timeout=SOCKET_TIMEOUT_SECONDS)
        except HTTPError as error:
            error.close()
            raise ServingSmokeError("http") from None
        with response:
            if response.status != 200:
                raise ServingSmokeError("http")
            body = bytearray()
            while len(body) <= MAX_RESPONSE_BYTES:
                chunk = response.read(MAX_RESPONSE_BYTES + 1 - len(body))
                if not chunk:
                    break
                body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ServingSmokeError("response_too_large")
    except (OSError, http.client.HTTPException):
        raise ServingSmokeError("transport") from None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):
        raise ServingSmokeError("schema") from None
    if not isinstance(payload, dict):
        raise ServingSmokeError("schema")
    return payload


def _validate_metadata(payload: dict[str, object], *, health: bool = False) -> None:
    fields = ("model_id", "model_version", "run_id")
    if health:
        fields += ("registered_model", "alias")
    if payload.get("model_id") != "invoice-review" or any(
        not isinstance(payload.get(field), str) or not payload[field].strip() for field in fields
    ):
        raise ServingSmokeError("schema")


def run_serving_smoke() -> dict[str, object]:
    """GET health, then POST one synthetic prediction; preserve independent snapshots.

    MODEL_API_URL is mandatory. No redirects, ambient proxies, credentials, cookies,
    or retries are used. Each response is limited to 64 KiB; the 10-second socket
    timeout is not a total deadline. Failures expose only a safe category.
    """
    base = _base_url()
    opener = build_opener(ProxyHandler({}), _NoRedirects())
    health = _request_json(opener, Request(base + "/models/invoice-review/health"))
    _validate_metadata(health, health=True)
    features = {
        "invoice_amount_cents": 125000,
        "vendor_tenure_days": 365,
        "previous_incidents_12m": 0,
        "amount_vs_vendor_median": 1.2,
        "has_purchase_order": True,
        "three_way_match": True,
        "bank_account_recently_changed": False,
        "country_risk": "low",
    }
    prediction = _request_json(
        opener,
        Request(
            base + "/models/invoice-review/predict",
            data=json.dumps(features).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
    )
    _validate_metadata(prediction)
    probability = prediction.get("probability")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
        or not math.isfinite(probability)
    ):
        raise ServingSmokeError("schema")
    warnings = []
    if any(health[field] != prediction[field] for field in ("model_version", "run_id")):
        warnings.append(
            "Model snapshot changed between health and prediction; calls are not atomic."
        )
    return {"health": health, "prediction": prediction, "warnings": warnings}
