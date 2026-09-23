"""Pinned TypeSafe Jev predictions with validated, resumable local response caches.

The cache contains comment text and model responses, never API credentials.
Network access must be explicitly enabled; offline replay is the default.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeGuard
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from review_router.data import LABELS

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
CACHE_VERSION = 1
MAX_RESPONSE_BYTES = 1_000_000


class JevError(RuntimeError):
    """An API or cache failure that prevents complete evaluation."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise JevError("Jev endpoint redirected; request stopped to protect credentials.")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _nonnegative_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _validate_questions(questions: Any) -> dict[str, Any]:
    if not isinstance(questions, dict) or set(questions) != set(LABELS):
        raise ValueError("Jev questions must contain exactly the six Jigsaw labels.")
    for label in LABELS:
        question = questions[label]
        if not isinstance(question, dict) or question.get("type") != "noul":
            raise ValueError(f"Jev question {label} must have type noul.")
        instructions = question.get("instructions")
        criteria = question.get("criteria")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"Jev question {label} needs instructions.")
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError(f"Jev question {label} needs true and false criteria.")
        if any(not isinstance(v, str) or not v.strip() for v in criteria.values()):
            raise ValueError(f"Jev question {label} criteria must be nonempty text.")
    # Copy nested structures so later caller mutations cannot change the protocol.
    return dict(json.loads(_canonical(questions)))


def load_questions(path: str | Path) -> dict[str, Any]:
    """Load a six-question JSON mapping, without accepting an alternate label order."""
    return _validate_questions(json.loads(Path(path).read_text(encoding="utf-8")))


class JevClient:
    def __init__(
        self,
        model: str,
        questions: dict[str, Any],
        cache_dir: str | Path,
        api_key: str | None = None,
        workers: int = 2,
        endpoint: str = DEFAULT_ENDPOINT,
        requests_per_second: float = 2.0,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not re.fullmatch(r"jev-\d+\.\d+\.\d+", model):
            raise ValueError(
                "Pin a full Jev version, for example jev-1.13.0; aliases are forbidden."
            )
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Jev endpoint must be an HTTPS URL without credentials or query.")
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
            raise ValueError("Jev workers must be an integer between 1 and 16.")
        if not _nonnegative_number(requests_per_second) or requests_per_second == 0:
            raise ValueError("Jev requests_per_second must be finite and positive.")
        if not _nonnegative_number(timeout) or timeout == 0:
            raise ValueError("Jev timeout must be finite and positive.")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("Jev max_retries must be an integer between 0 and 8.")
        if not 0 <= max_retries <= 8:
            raise ValueError("Jev max_retries must be an integer between 0 and 8.")
        self.model = model
        self.questions = _validate_questions(questions)
        self.cache_dir = Path(cache_dir)
        self.endpoint = endpoint
        self.workers = workers
        self.requests_per_second = requests_per_second
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = api_key
        self._lock = threading.Lock()
        self._next_request = 0.0
        self._opener = build_opener(_NoRedirect())
        self._counts: dict[str, int] = {
            "requested_rows": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "network_calls": 0,
            "network_successes": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cached_output_tokens": 0,
        }
        self._latencies: list[float] = []
        self._network_seconds = 0.0

    def _request_binding(self, text: str) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "body": {
                "model": self.model,
                "state": {"comment_text": text},
                "questions": self.questions,
            },
        }

    def cache_key(self, text: str) -> str:
        """Hash the exact model, endpoint, questions and comment text."""
        return _digest(self._request_binding(text))

    def _validated_response(self, raw: Any) -> tuple[list[float], dict[str, int]]:
        if not isinstance(raw, dict) or raw.get("model") != self.model:
            raise JevError("Jev response model does not match the pinned version.")
        answers = raw.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(LABELS):
            raise JevError("Jev response must contain exactly the six requested answers.")
        probabilities = []
        for label in LABELS:
            answer = answers[label]
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise JevError(f"Jev response for {label} is not a noul answer.")
            value = answer.get("noul")
            if not _nonnegative_number(value) or value > 1:
                raise JevError(f"Jev probability for {label} must be finite and in [0, 1].")
            probabilities.append(float(value))
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            raise JevError("Jev response is missing token usage.")
        tokens: dict[str, int] = {}
        for name in ("input_tokens", "output_tokens"):
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise JevError(f"Jev response has invalid {name}.")
            tokens[name] = value
        return probabilities, tokens

    def _read_cache(self, path: Path, binding: dict[str, Any]) -> list[float] | None:
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(record, dict)
                or record.get("cache_version") != CACHE_VERSION
                or record.get("request") != binding
                or record.get("request_sha256") != _digest(binding)
            ):
                raise JevError("Jev cached response does not match its request.")
            latency = record.get("elapsed_seconds")
            if not _nonnegative_number(latency):
                raise JevError("Jev cached response has invalid elapsed time.")
            probabilities, usage = self._validated_response(record.get("response"))
        except (OSError, ValueError, TypeError):
            raise JevError(
                "Jev cached response is unreadable or malformed; replay stopped."
            ) from None
        with self._lock:
            self._counts["cache_hits"] += 1
            for name, count in usage.items():
                self._counts["cached_" + name] += count
        return probabilities

    def _write_cache(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=".jev-", delete=False
            ) as stream:
                temporary = stream.name
                stream.write(_canonical(record))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request)
            self._next_request = scheduled + 1.0 / self.requests_per_second
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    def _fetch(self, binding: dict[str, Any], key: str) -> tuple[dict[str, Any], float]:
        credential = self._api_key or os.environ.get("TYPESAFE_API_KEY")
        if not credential or not credential.strip():
            raise JevError("Set TYPESAFE_API_KEY to enable uncached Jev requests.")
        if "\n" in credential or "\r" in credential:
            raise JevError("Jev API credential contains invalid characters.")
        request = Request(
            self.endpoint,
            data=_canonical(binding["body"]).encode("utf-8"),
            headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            self._throttle()
            started = time.monotonic()
            with self._lock:
                self._counts["network_calls"] += 1
            retry = False
            status = "transport failure"
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise JevError("Jev response exceeded the allowed size.")
                try:
                    raw = json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    raise JevError("Jev returned malformed JSON; response body omitted.") from None
                self._validated_response(raw)
                # A reflected credential must never reach the response cache.
                if credential in _canonical(raw):
                    raise JevError("Jev returned credential material; response discarded.")
                return dict(raw), time.monotonic() - started
            except HTTPError as error:
                status = f"HTTP {error.code}"
                retry = error.code == 429 or 500 <= error.code < 600
                error.close()
            except (URLError, TimeoutError, ConnectionError, OSError):
                retry = True
            finally:
                with self._lock:
                    self._network_seconds += time.monotonic() - started
            if not retry or attempt == self.max_retries:
                raise JevError(
                    f"Jev request {key[:12]} failed ({status}); no rows were omitted. "
                    "Response details and credentials are not logged."
                ) from None
            with self._lock:
                self._counts["retries"] += 1
            time.sleep(min(2.0**attempt, 8.0))
        raise AssertionError("unreachable retry state")

    def _predict_one(self, text: str, allow_network: bool) -> list[float]:
        binding = self._request_binding(text)
        key = _digest(binding)
        path = self.cache_dir / f"{key}.json"
        cached = self._read_cache(path, binding)
        if cached is not None:
            return cached
        with self._lock:
            self._counts["cache_misses"] += 1
        if not allow_network:
            raise JevError(f"Jev cache missing for request {key[:12]}; offline replay stopped.")
        raw, elapsed = self._fetch(binding, key)
        probabilities, usage = self._validated_response(raw)
        self._write_cache(
            path,
            {
                "cache_version": CACHE_VERSION,
                "request_sha256": key,
                "request": binding,
                "response": raw,
                "elapsed_seconds": elapsed,
            },
        )
        with self._lock:
            self._counts["network_successes"] += 1
            self._latencies.append(elapsed)
            for name, count in usage.items():
                self._counts[name] += count
        return probabilities

    def predict(self, texts: list[str], *, allow_network: bool = False) -> np.ndarray:
        """Return every row in LABELS order, or raise without returning partial data.

        Exact duplicate comments share one prediction and one billable request.
        Cache hit and miss counts refer to unique comments within each invocation.
        """
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Jev input must contain only text strings.")
        with self._lock:
            self._counts["requested_rows"] += len(texts)
        if not texts:
            return np.empty((0, len(LABELS)), dtype=float)
        unique = list(dict.fromkeys(texts))
        pool = ThreadPoolExecutor(max_workers=self.workers)
        futures = {pool.submit(self._predict_one, text, allow_network): text for text in unique}
        try:
            rows = {futures[future]: future.result() for future in as_completed(futures)}
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        return np.asarray([rows[text] for text in texts], dtype=float)

    def summary(self) -> dict[str, Any]:
        """Report observed usage; unknown monetary charges are not estimated."""
        with self._lock:
            latencies = list(self._latencies)
            counts = dict(self._counts)
            seconds = self._network_seconds
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "questions_sha256": _digest(self.questions),
            **counts,
            "total_tokens": counts["input_tokens"] + counts["output_tokens"],
            "network_seconds": seconds,
            "mean_latency_seconds": float(np.mean(latencies)) if latencies else None,
            "p50_latency_seconds": float(np.quantile(latencies, 0.5)) if latencies else None,
            "p95_latency_seconds": float(np.quantile(latencies, 0.95)) if latencies else None,
            "billing_usd": None,
            "billing_note": (
                "Token usage covers validated successful responses in this process. "
                "Failed attempts may incur additional unreported charges; cached tokens "
                "are reported separately. No price schedule was assumed."
            ),
        }
