from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from review_router.data import LABELS
from review_router.jev import JevClient, JevError, load_questions

MODEL = "jev-1.13.0"
SECRET = "secret-api-key-never-log"
QUESTIONS = Path(__file__).resolve().parents[1] / "configs" / "jev_questions.json"


def _response(value: float = 0.2) -> dict[str, Any]:
    return {
        "model": MODEL,
        "answers": {label: {"type": "noul", "noul": value} for label in LABELS},
        "usage": {"input_tokens": 100, "output_tokens": 6},
    }


class FakeOpener:
    def __init__(self, events: list[Any]) -> None:
        self.events = iter(events)
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float) -> io.BytesIO:
        self.requests.append(request)
        event = next(self.events)
        if isinstance(event, Exception):
            raise event
        data = event if isinstance(event, bytes) else json.dumps(event).encode()
        return io.BytesIO(data)


def _client(tmp_path: Path, **kwargs: Any) -> JevClient:
    return JevClient(MODEL, load_questions(QUESTIONS), tmp_path, workers=1, **kwargs)


@pytest.fixture(autouse=True)
def _no_credentials_or_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("review_router.jev.time.sleep", lambda delay: None)


def test_network_cache_and_offline_replay_preserve_order(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET)
    opener = FakeOpener([_response(0.1), _response(0.9)])
    client._opener = opener
    actual = client.predict(["first", "second", "first"], allow_network=True)
    np.testing.assert_array_equal(actual[:, 0], [0.1, 0.9, 0.1])
    assert actual.shape == (3, 6)
    assert len(opener.requests) == 2
    body = json.loads(opener.requests[0].data)
    assert body == {
        "model": MODEL,
        "state": {"comment_text": "first"},
        "questions": load_questions(QUESTIONS),
    }
    assert opener.requests[0].get_header("Authorization") == "Bearer " + SECRET
    summary = client.summary()
    assert summary["requested_rows"] == 3
    assert summary["network_successes"] == 2
    assert summary["input_tokens"] == 200
    assert summary["output_tokens"] == 12
    assert summary["billing_usd"] is None
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob(".jev-*"))
    assert all(SECRET not in path.read_text() for path in tmp_path.glob("*.json"))
    assert SECRET not in json.dumps(summary)

    replay = _client(tmp_path)
    replay._opener = FakeOpener([])
    np.testing.assert_array_equal(replay.predict(["second", "first"])[:, 0], [0.9, 0.1])
    assert replay.summary()["cache_hits"] == 2
    assert replay.summary()["cached_input_tokens"] == 200
    assert replay.summary()["input_tokens"] == 0
    assert replay.summary()["network_calls"] == 0


def test_empty_input_has_six_columns_without_network(tmp_path: Path) -> None:
    assert _client(tmp_path).predict([]).shape == (0, 6)


def test_probability_columns_follow_labels_not_response_order(tmp_path: Path) -> None:
    response = _response()
    response["answers"] = {
        label: {"type": "noul", "noul": i / 10} for i, label in reversed(list(enumerate(LABELS)))
    }
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([response])
    np.testing.assert_array_equal(
        client.predict(["first"], allow_network=True)[0], np.arange(6) / 10
    )


def test_network_opt_in_and_credentials_are_both_required(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET)
    with pytest.raises(JevError, match="offline replay stopped"):
        client.predict(["missing"])
    assert client.summary()["network_calls"] == 0
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        _client(tmp_path).predict(["missing"], allow_network=True)
    assert not list(tmp_path.glob("*.json"))


def test_cache_is_bound_to_text_version_endpoint_and_questions(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([_response()])
    client.predict(["original"], allow_network=True)
    questions = load_questions(QUESTIONS)
    questions["toxic"]["instructions"] += " Different definition."
    candidates = [
        JevClient("jev-1.13.1", load_questions(QUESTIONS), tmp_path),
        JevClient(MODEL, questions, tmp_path),
        JevClient(
            MODEL,
            load_questions(QUESTIONS),
            tmp_path,
            endpoint="https://alternate.typesafe.ai/v1/systemone",
        ),
    ]
    for candidate in candidates:
        with pytest.raises(JevError, match="cache missing"):
            candidate.predict(["original"])
    with pytest.raises(JevError, match="cache missing"):
        client.predict(["changed"])
    question_copy = load_questions(QUESTIONS)
    frozen = JevClient(MODEL, question_copy, tmp_path)
    original_key = frozen.cache_key("original")
    question_copy["toxic"]["instructions"] = "mutated"
    assert frozen.cache_key("original") == original_key


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cache_version", 999),
        ("request_sha256", "incorrect"),
        ("request", {}),
        ("elapsed_seconds", float("nan")),
        ("response", {"model": "jev-1.13.1"}),
    ],
)
def test_corrupted_cache_stops_even_when_network_allowed(
    tmp_path: Path, field: str, replacement: Any
) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([_response()])
    client.predict(["first"], allow_network=True)
    path = tmp_path / (client.cache_key("first") + ".json")
    record = json.loads(path.read_text())
    record[field] = replacement
    path.write_text(json.dumps(record))
    replay = _client(tmp_path, api_key=SECRET)
    replay._opener = FakeOpener([])
    with pytest.raises(JevError):
        replay.predict(["first"], allow_network=True)
    assert replay.summary()["network_calls"] == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, None, "0.4"])
def test_invalid_probabilities_cannot_be_cached(tmp_path: Path, value: Any) -> None:
    response = _response()
    response["answers"]["toxic"]["noul"] = value
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([response])
    with pytest.raises(JevError, match="probability"):
        client.predict(["first"], allow_network=True)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize(
    "response",
    [
        {"model": "jev-1.13.1"},
        {**_response(), "answers": {}},
        {**_response(), "usage": {"input_tokens": True, "output_tokens": 6}},
        {**_response(), "usage": None},
        {**_response(), "answers": {**_response()["answers"], "extra": {}}},
        {**_response(), "answers": {**_response()["answers"], "toxic": {"type": "choice"}}},
    ],
)
def test_response_contract_and_model_drift_fail_closed(tmp_path: Path, response: Any) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([response])
    with pytest.raises(JevError):
        client.predict(["first"], allow_network=True)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("error", [401, 403, 400])
def test_nonretryable_http_errors_are_redacted(tmp_path: Path, error: int) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener(
        [HTTPError("https://example.com/" + SECRET, error, SECRET, {}, io.BytesIO(SECRET.encode()))]
    )
    with pytest.raises(JevError, match=f"HTTP {error}") as caught:
        client.predict(["first"], allow_network=True)
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value)
    assert client.summary()["network_calls"] == 1
    assert not list(tmp_path.glob("*.json"))


def test_finite_retry_handles_rate_limits_server_and_transport_errors(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener(
        [
            HTTPError("https://example.com", 429, SECRET, {}, io.BytesIO()),
            HTTPError("https://example.com", 503, SECRET, {}, io.BytesIO()),
            URLError(SECRET),
            _response(),
        ]
    )
    assert client.predict(["first"], allow_network=True)[0, 0] == 0.2
    assert client.summary()["network_calls"] == 4
    assert client.summary()["retries"] == 3


def test_retry_exhaustion_fails_without_cache_or_secret_leak(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET, max_retries=1)
    client._opener = FakeOpener([TimeoutError(SECRET), URLError(SECRET)])
    with pytest.raises(JevError, match="transport failure") as caught:
        client.predict(["first"], allow_network=True)
    assert SECRET not in str(caught.value)
    assert client.summary()["network_calls"] == 2
    assert not list(tmp_path.glob("*.json"))


def test_completed_requests_survive_interruption_and_resume(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET, max_retries=0)
    client._opener = FakeOpener([_response(0.1), TimeoutError("broken")])
    with pytest.raises(JevError):
        client.predict(["first", "second"], allow_network=True)
    assert len(list(tmp_path.glob("*.json"))) == 1
    resumed = _client(tmp_path, api_key=SECRET)
    resumed._opener = FakeOpener([_response(0.9)])
    np.testing.assert_array_equal(
        resumed.predict(["first", "second"], allow_network=True)[:, 0], [0.1, 0.9]
    )
    assert resumed.summary()["cache_hits"] == 1
    assert resumed.summary()["network_calls"] == 1


@pytest.mark.parametrize("payload", [b"secret-api-key-never-log", b"{"])
def test_invalid_json_error_never_quotes_response(tmp_path: Path, payload: bytes) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([payload])
    with pytest.raises(JevError, match="malformed JSON") as caught:
        client.predict(["first"], allow_network=True)
    assert SECRET not in str(caught.value)


def test_reflected_secret_is_discarded(tmp_path: Path) -> None:
    client = _client(tmp_path, api_key=SECRET)
    client._opener = FakeOpener([{**_response(), "unexpected_echo": SECRET}])
    with pytest.raises(JevError, match="credential material"):
        client.predict(["first"], allow_network=True)
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("model", ["latest", "jev-latest", "jev-1", "jev-1.13"])
def test_unpinned_models_are_forbidden(tmp_path: Path, model: str) -> None:
    with pytest.raises(ValueError, match="Pin"):
        JevClient(model, load_questions(QUESTIONS), tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": 0},
        {"workers": 17},
        {"workers": True},
        {"requests_per_second": 0},
        {"requests_per_second": float("nan")},
        {"timeout": 0},
        {"max_retries": -1},
        {"max_retries": 9},
        {"endpoint": "http://api.typesafe.ai/v1/systemone"},
        {"endpoint": "https://user:secret@api.typesafe.ai/v1/systemone"},
        {"endpoint": "https://api.typesafe.ai/v1/systemone?key=secret"},
    ],
)
def test_invalid_client_configuration_is_rejected(tmp_path: Path, kwargs: Any) -> None:
    with pytest.raises(ValueError):
        JevClient(MODEL, load_questions(QUESTIONS), tmp_path, **kwargs)


def test_rate_limit_reserves_separate_start_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("review_router.jev.time.monotonic", lambda: 10.0)
    monkeypatch.setattr("review_router.jev.time.sleep", delays.append)
    client = _client(tmp_path, requests_per_second=2)
    for _ in range(3):
        client._throttle()
    assert delays == [0.5, 1.0]
