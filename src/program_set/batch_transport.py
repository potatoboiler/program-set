"""A small, app-owned transport for provider batch requests.

This module stores only batch request payloads and transport metadata/results. It
does not read or persist Strands conversation or graph state.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypedDict, cast

from program_set.batch_json import canonical_json, field, jsonable
from program_set.batch_result import (
    BatchResult,
    BatchResultError,
    BatchResultType,
    NormalizedContentBlock,
    StopReason,
    parse_batch_result,
)

RequestStatus = Literal[
    "queued",
    "submitting",
    "submit_unknown",
    "submitted",
    "succeeded",
    "errored",
    "canceled",
    "expired",
]


class BatchRequestEnvelope(TypedDict):
    """Correlation envelope around an opaque provider-specific request payload."""

    custom_id: str
    params: dict[str, object]


@dataclass(frozen=True)
class BatchInfo:
    batch_id: str
    processing_status: str


@dataclass(frozen=True)
class EnqueuedRequest:
    request_id: str
    custom_id: str


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    custom_id: str
    metadata: Mapping[str, str]
    status: RequestStatus
    batch_id: str | None
    result_status: BatchResultType | None
    result: BatchResult | None
    submission_id: str | None


class BatchClient(Protocol):
    """Provider adapter with a side-effect-free local validation preflight."""

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> None: ...

    def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo: ...

    def retrieve_batch(self, batch_id: str) -> BatchInfo: ...

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> Iterable[object]: ...


class _AnthropicBatchesResource(Protocol):
    def create(self, *, requests: Sequence[Any]) -> object: ...

    def retrieve(self, batch_id: str) -> object: ...

    def results(self, batch_id: str) -> Iterable[Any]: ...


class _AnthropicMessagesResource(Protocol):
    @property
    def batches(self) -> _AnthropicBatchesResource: ...


class _AnthropicSDKClient(Protocol):
    @property
    def messages(self) -> _AnthropicMessagesResource: ...

    def with_options(self, *, max_retries: int) -> _AnthropicSDKClient: ...


class AnthropicBatchClient:
    """Adapt an injected Anthropic SDK client to :class:`BatchClient`.

    The SDK client is injected by the application, so constructing this adapter
    never creates a client, reads credentials, or makes a request.

    Per-call retries are disabled for batch creation so a POST retry after a
    timeout cannot create duplicate remote work.
    """

    def __init__(self, client: _AnthropicSDKClient) -> None:
        self._client = client

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> None:
        for request in requests:
            _validate_anthropic_request(request)

    def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo:
        self.validate_requests(requests)
        resource = self._client.with_options(max_retries=0).messages.batches
        response = resource.create(requests=list(requests))
        return _batch_info(response)

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        response = self._batch_resource().retrieve(batch_id)
        return _batch_info(response)

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> Iterable[object]:
        del expected_custom_ids
        for response in self._batch_resource().results(batch_id):
            custom_id = field(response, "custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise BatchProtocolError(
                    "Anthropic batch result has no valid custom_id"
                )
            yield {
                "custom_id": custom_id,
                "result": _normalize_anthropic_result(field(response, "result")),
            }

    def _batch_resource(self) -> _AnthropicBatchesResource:
        return self._client.messages.batches


class BatchTransportError(RuntimeError):
    """Base exception for transport and persisted-state errors."""


class RequestConflictError(BatchTransportError):
    """An idempotency key was reused for different request data."""


class UnknownRequestError(BatchTransportError):
    """The requested application request ID is not in this transport."""


class UnknownBatchError(BatchTransportError):
    """The batch ID is not registered in this transport."""


class BatchProtocolError(BatchTransportError):
    """The client returned a response inconsistent with the registered batch."""


class SubmissionUncertainError(BatchTransportError):
    """Batch creation failed after it may have reached the provider.

    Requests in ``submission_id`` are deliberately not automatically retried:
    a caller must establish that the create call was not accepted before
    invoking :meth:`BatchTransport.requeue_uncertain`.
    """

    def __init__(self, submission_id: str, request_ids: Sequence[str]) -> None:
        self.submission_id = submission_id
        self.request_ids = tuple(request_ids)
        super().__init__(
            "batch submission outcome is unknown for submission "
            f"{submission_id}; do not retry unless non-acceptance is confirmed"
        )


class BatchTransport:
    """Queue and reconcile complete provider batch requests.

    ``flush`` is the only submission trigger; callers choose when to call it.
    The SQLite file holds queued request payloads, correlation metadata, known
    batch IDs, and returned batch results. There is no timer or worker loop.
    """

    def __init__(self, client: BatchClient, database: str | Path) -> None:
        self._client = client
        self._lock = threading.RLock()
        database_path = str(Path(database).expanduser())
        if database_path != ":memory:":
            Path(database_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def enqueue(
        self,
        params: object,
        *,
        request_id: str | None = None,
        custom_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> EnqueuedRequest:
        """Persist an opaque provider request and return its stable IDs.

        If IDs are omitted, UUIDs are generated once and persisted. By default
        ``custom_id`` equals ``request_id``; supplying either ID lets callers
        correlate results with their own durable work identifiers. Both IDs
        must be non-empty; provider-specific ID constraints belong to adapters.
        Re-enqueueing identical data under the same request ID is idempotent.
        """

        payload = _validated_params(params)
        if request_id is None:
            request_id = str(uuid.uuid4())
        if custom_id is None:
            custom_id = request_id
        _validate_id(request_id, "request_id")
        _validate_id(custom_id, "custom_id")
        request_metadata: dict[str, str] = {}
        for key, value in (metadata or {}).items():
            if not isinstance(value, str):
                raise TypeError("metadata must map strings to strings")
            request_metadata[key] = value

        payload_json = _dump_json(payload)
        metadata_json = _dump_json(request_metadata)
        created_at = _now()

        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT custom_id, params_json, metadata_json "
                "FROM requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["custom_id"] != custom_id
                    or existing["params_json"] != payload_json
                    or existing["metadata_json"] != metadata_json
                ):
                    raise RequestConflictError(
                        f"request_id {request_id!r} already has different data"
                    )
                return EnqueuedRequest(request_id, custom_id)

            try:
                self._connection.execute(
                    "INSERT INTO requests (request_id, custom_id, params_json, "
                    "metadata_json, status, created_at) "
                    "VALUES (?, ?, ?, ?, 'queued', ?)",
                    (
                        request_id,
                        custom_id,
                        payload_json,
                        metadata_json,
                        created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RequestConflictError(
                    f"custom_id {custom_id!r} is already registered"
                ) from exc
        return EnqueuedRequest(request_id, custom_id)

    def flush(self) -> BatchInfo | None:
        """Submit all currently queued requests as one remote batch.

        A failed create call leaves the requests in ``submit_unknown``. This is
        intentional: blindly replaying a request after a timeout can create
        duplicate remote work. See :meth:`requeue_uncertain` for the explicit,
        operator-confirmed recovery path.
        """

        with self._lock:
            rows = self._connection.execute(
                "SELECT request_id, custom_id, params_json FROM requests "
                "WHERE status = 'queued' ORDER BY created_at, request_id"
            ).fetchall()
            if not rows:
                return None

            submission_id = str(uuid.uuid4())
            request_ids = [str(row["request_id"]) for row in rows]
            requests: list[BatchRequestEnvelope] = [
                {
                    "custom_id": str(row["custom_id"]),
                    "params": cast(dict[str, object], json.loads(row["params_json"])),
                }
                for row in rows
            ]
            # Provider validation must fail before the durable state suggests
            # that the create request may have reached the provider.
            self._client.validate_requests(requests)
            with self._connection:
                self._connection.executemany(
                    "UPDATE requests SET status = 'submitting', submission_id = ? "
                    "WHERE request_id = ? AND status = 'queued'",
                    [(submission_id, request_id) for request_id in request_ids],
                )

            try:
                batch = self._client.create_batch(requests)
                _validate_batch_info(batch)
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO batches (batch_id, processing_status, updated_at) "
                        "VALUES (?, ?, ?)",
                        (batch.batch_id, batch.processing_status, _now()),
                    )
                    self._connection.executemany(
                        "UPDATE requests SET status = 'submitted', batch_id = ? "
                        "WHERE request_id = ? AND status = 'submitting' "
                        "AND submission_id = ?",
                        [
                            (batch.batch_id, request_id, submission_id)
                            for request_id in request_ids
                        ],
                    )
            except Exception as exc:
                with self._connection:
                    self._connection.execute(
                        "UPDATE requests SET status = 'submit_unknown' "
                        "WHERE status = 'submitting' AND submission_id = ?",
                        (submission_id,),
                    )
                raise SubmissionUncertainError(submission_id, request_ids) from exc
            return batch

    def poll(self, batch_id: str) -> BatchInfo:
        """Refresh a registered batch and persist terminal results by custom ID."""

        with self._lock:
            known = self._connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if known is None:
                raise UnknownBatchError(f"batch_id {batch_id!r} is not registered")

            batch = self._client.retrieve_batch(batch_id)
            _validate_batch_info(batch)
            if batch.batch_id != batch_id:
                raise BatchProtocolError(
                    "retrieved batch "
                    f"{batch.batch_id!r} for requested batch {batch_id!r}"
                )
            with self._connection:
                self._connection.execute(
                    "UPDATE batches SET processing_status = ?, updated_at = ? "
                    "WHERE batch_id = ?",
                    (batch.processing_status, _now(), batch_id),
                )

            if batch.processing_status == "ended":
                self._persist_results(batch_id)
            return batch

    def _persist_results(self, batch_id: str) -> None:
        registered = {
            str(row["custom_id"]): str(row["request_id"])
            for row in self._connection.execute(
                "SELECT request_id, custom_id FROM requests WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        }
        results_by_custom_id: dict[str, tuple[BatchResultType, str]] = {}
        for response in self._client.iter_batch_results(
            batch_id, expected_custom_ids=tuple(registered)
        ):
            custom_id = field(response, "custom_id")
            result = field(response, "result")
            if not isinstance(custom_id, str) or not custom_id:
                raise BatchProtocolError("batch result has no valid custom_id")
            if custom_id in results_by_custom_id:
                raise BatchProtocolError(
                    f"duplicate result for custom_id {custom_id!r}"
                )
            try:
                normalized_result = parse_batch_result(jsonable(result))
            except BatchResultError as exc:
                raise BatchProtocolError(
                    f"invalid result for {custom_id!r}: {exc}"
                ) from exc
            result_status = normalized_result["type"]
            results_by_custom_id[custom_id] = (
                result_status,
                _dump_json(normalized_result),
            )

        with self._connection:
            for custom_id, (result_status, result_json) in results_by_custom_id.items():
                request_id = registered.get(custom_id)
                if request_id is None:
                    raise BatchProtocolError(
                        f"result custom_id {custom_id!r} does not belong to "
                        f"batch {batch_id!r}"
                    )
                self._connection.execute(
                    "UPDATE requests SET status = ?, result_status = ?, "
                    "result_json = ? "
                    "WHERE request_id = ? AND batch_id = ?",
                    (result_status, result_status, result_json, request_id, batch_id),
                )
            missing = registered.keys() - results_by_custom_id.keys()
            if missing:
                raise BatchProtocolError(
                    f"batch {batch_id!r} results omitted custom_ids {sorted(missing)!r}"
                )

    def get_request(self, request_id: str) -> RequestRecord:
        """Return a request's durable transport state and, if available, result."""

        with self._lock:
            row = self._connection.execute(
                "SELECT request_id, custom_id, metadata_json, status, batch_id, "
                "result_status, result_json, submission_id FROM requests "
                "WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            raise UnknownRequestError(f"request_id {request_id!r} is not registered")
        raw_result = (
            json.loads(row["result_json"]) if row["result_json"] is not None else None
        )
        try:
            result = parse_batch_result(raw_result) if raw_result is not None else None
        except BatchResultError as exc:
            raise BatchProtocolError(
                f"stored result for {request_id!r} is malformed: {exc}"
            ) from exc
        return RequestRecord(
            request_id=str(row["request_id"]),
            custom_id=str(row["custom_id"]),
            metadata=cast(dict[str, str], json.loads(row["metadata_json"])),
            status=cast(RequestStatus, row["status"]),
            batch_id=row["batch_id"],
            result_status=cast(BatchResultType | None, row["result_status"]),
            result=result,
            submission_id=row["submission_id"],
        )

    def list_batches(self) -> tuple[BatchInfo, ...]:
        """List locally registered remote batches, including after reopening DB."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT batch_id, processing_status FROM batches "
                "ORDER BY updated_at, batch_id"
            ).fetchall()
        return tuple(
            BatchInfo(str(row["batch_id"]), str(row["processing_status"]))
            for row in rows
        )

    def requeue_uncertain(self, submission_id: str) -> int:
        """Explicitly requeue one uncertain submission after non-acceptance is known.

        This operation updates the entire original create call together. It does
        not query the provider or decide whether a retry is safe.
        """

        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT request_id, status FROM requests WHERE submission_id = ?",
                (submission_id,),
            ).fetchall()
            if not rows or any(row["status"] != "submit_unknown" for row in rows):
                raise BatchTransportError(
                    f"submission {submission_id!r} is not wholly uncertain"
                )
            self._connection.execute(
                "UPDATE requests SET status = 'queued', submission_id = NULL "
                "WHERE submission_id = ?",
                (submission_id,),
            )
            return len(rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    processing_status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    custom_id TEXT NOT NULL UNIQUE,
                    params_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    batch_id TEXT REFERENCES batches(batch_id),
                    result_status TEXT,
                    result_json TEXT,
                    submission_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS requests_status_created
                    ON requests(status, created_at, request_id);
                CREATE INDEX IF NOT EXISTS requests_submission
                    ON requests(submission_id);
                """
            )
            # A process that died during create may have sent the remote call.
            # Such rows must never be silently treated as fresh queued work.
            self._connection.execute(
                "UPDATE requests SET status = 'submit_unknown' "
                "WHERE status = 'submitting'"
            )


def _validated_params(params: object) -> dict[str, object]:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")
    decoded = json.loads(_dump_json(dict(params)))
    if not isinstance(decoded, dict):
        raise TypeError("params must serialize as an object")
    result: dict[str, object] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise TypeError("params keys must be strings")
        result[key] = value
    return result


def _validate_anthropic_request(request: BatchRequestEnvelope) -> None:
    custom_id = request["custom_id"]
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", custom_id) is None:
        raise ValueError(
            "Anthropic custom_id must contain 1-64 ASCII letters, digits, "
            "underscores, or hyphens"
        )

    params = request["params"]
    model = params.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Anthropic request model must be a non-empty string")
    max_tokens = params.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("Anthropic max_tokens must be a positive integer")
    messages = params.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Anthropic messages must be a non-empty list")
    if params.get("stream") is True:
        raise ValueError("stream=True is not supported in Anthropic batch requests")
    if "speed" in params:
        raise ValueError("speed is not supported in Anthropic batch requests")


def _validate_id(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _batch_info(value: object) -> BatchInfo:
    batch_id = field(value, "id")
    processing_status = field(value, "processing_status")
    if not isinstance(batch_id, str) or not isinstance(processing_status, str):
        raise BatchProtocolError("Anthropic batch response is missing id/status")
    return BatchInfo(batch_id, processing_status)


def _normalize_anthropic_result(value: object) -> BatchResult:
    result_type = field(value, "type")
    if result_type == "succeeded":
        message = field(value, "message")
        content = field(message, "content")
        stop_reason = field(message, "stop_reason")
        if not isinstance(content, list):
            raise BatchProtocolError("Anthropic result message content is not a list")
        if stop_reason == "stop_sequence":
            stop_reason = "end_turn"
        normalized_stop_reason: StopReason
        if stop_reason == "end_turn":
            normalized_stop_reason = "end_turn"
        elif stop_reason == "tool_use":
            normalized_stop_reason = "tool_use"
        elif stop_reason == "max_tokens":
            normalized_stop_reason = "max_tokens"
        else:
            raise BatchProtocolError(
                f"unsupported Anthropic stop reason {stop_reason!r}"
            )

        normalized_content: list[NormalizedContentBlock] = []
        for block in content:
            block_type = field(block, "type")
            if block_type == "text":
                text = field(block, "text")
                if not isinstance(text, str):
                    raise BatchProtocolError("Anthropic text block is not text")
                normalized_content.append({"type": "text", "text": text})
            elif block_type == "tool_use":
                tool_id = field(block, "id")
                name = field(block, "name")
                tool_input = field(block, "input")
                if (
                    not isinstance(tool_id, str)
                    or not isinstance(name, str)
                    or not isinstance(tool_input, Mapping)
                ):
                    raise BatchProtocolError("Anthropic tool_use block is malformed")
                normalized_input: dict[str, object] = {}
                for key, item in tool_input.items():
                    if not isinstance(key, str):
                        raise BatchProtocolError(
                            "Anthropic tool_use input is not a string-keyed object"
                        )
                    normalized_input[key] = item
                normalized_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": normalized_input,
                    }
                )
            elif block_type == "reasoning":
                signature = field(block, "signature")
                if not isinstance(signature, str):
                    raise BatchProtocolError("Anthropic reasoning block is malformed")
                normalized_content.append({"type": "reasoning", "signature": signature})
            else:
                raise BatchProtocolError(
                    f"unsupported Anthropic result content block {block_type!r}"
                )
        return {
            "type": "succeeded",
            "message": {
                "content": normalized_content,
                "stop_reason": normalized_stop_reason,
            },
        }
    if result_type == "errored":
        return {"type": "errored", "error": jsonable(field(value, "error"))}
    if result_type == "canceled":
        return {"type": "canceled"}
    if result_type == "expired":
        return {"type": "expired"}
    raise BatchProtocolError(f"unsupported Anthropic result type {result_type!r}")


def _validate_batch_info(batch: BatchInfo) -> None:
    if not batch.batch_id or not batch.processing_status:
        raise BatchProtocolError("batch response is missing id/status")


def _dump_json(value: object) -> str:
    return canonical_json(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()
