"""A small, app-owned transport for provider batch requests.

This module stores only batch request payloads and transport metadata/results. It
does not read or persist Strands conversation or graph state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self, TypedDict, cast

from program_set.batch_json import canonical_json, field, jsonable
from program_set.batch_result import (
    BatchResult,
    BatchResultError,
    BatchResultType,
    parse_batch_result,
)
from program_set.provider_profile import ProviderProfile

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


class BatchTransportError(RuntimeError):
    """Base exception for transport and persisted-state errors."""


class RequestConflictError(BatchTransportError):
    """An idempotency key was reused for different request data."""


class UnknownRequestError(BatchTransportError):
    """The requested application request ID is not in this transport."""


class UnknownBatchError(BatchTransportError):
    """The batch ID is not registered in this transport."""


class ProviderProfileNotFoundError(BatchTransportError):
    """The requested provider profile is not stored in this transport."""


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

    def __init__(self, client: BatchClient | None, database: str | Path) -> None:
        self._client = client
        self._lock = threading.RLock()
        database_path = str(Path(database).expanduser())
        if database_path != ":memory:":
            Path(database_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def set_client(self, client: BatchClient) -> None:
        """Bind a provider adapter, e.g. after loading its durable profile."""

        with self._lock:
            self._client = client

    def save_provider_profile(self, profile: ProviderProfile) -> None:
        """Insert or replace non-secret provider configuration by profile ID."""

        config_json = _dump_json(dict(profile.config))
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO provider_profiles (profile_id, provider, config_json) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(profile_id) DO UPDATE SET "
                "provider = excluded.provider, config_json = excluded.config_json",
                (profile.profile_id, profile.provider, config_json),
            )

    def get_provider_profile(self, profile_id: str) -> ProviderProfile:
        """Load and validate a provider profile from SQLite."""

        with self._lock:
            row = self._connection.execute(
                "SELECT profile_id, provider, config_json FROM provider_profiles "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
        if row is None:
            raise ProviderProfileNotFoundError(
                f"provider profile {profile_id!r} is not stored"
            )
        try:
            config = json.loads(row["config_json"])
            if not isinstance(config, Mapping):
                raise ValueError("provider profile config is not an object")
            return ProviderProfile(
                profile_id=str(row["profile_id"]),
                provider=row["provider"],
                config=config,
            )
        except (TypeError, ValueError) as exc:
            raise BatchProtocolError(
                f"stored provider profile {profile_id!r} is malformed: {exc}"
            ) from exc

    def list_provider_profiles(self) -> tuple[ProviderProfile, ...]:
        """List validated profiles in stable profile-ID order."""

        with self._lock:
            profile_ids = self._connection.execute(
                "SELECT profile_id FROM provider_profiles ORDER BY profile_id"
            ).fetchall()
        return tuple(
            self.get_provider_profile(str(row["profile_id"])) for row in profile_ids
        )

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

            client = self._require_client()

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
            client.validate_requests(requests)
            with self._connection:
                self._connection.executemany(
                    "UPDATE requests SET status = 'submitting', submission_id = ? "
                    "WHERE request_id = ? AND status = 'queued'",
                    [(submission_id, request_id) for request_id in request_ids],
                )

            try:
                batch = client.create_batch(requests)
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

            client = self._require_client()
            batch = client.retrieve_batch(batch_id)
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
                self._persist_results(batch_id, client)
            return batch

    def _persist_results(self, batch_id: str, client: BatchClient) -> None:
        registered = {
            str(row["custom_id"]): str(row["request_id"])
            for row in self._connection.execute(
                "SELECT request_id, custom_id FROM requests WHERE batch_id = ?",
                (batch_id,),
            ).fetchall()
        }
        results_by_custom_id: dict[str, tuple[BatchResultType, str]] = {}
        for response in client.iter_batch_results(
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
                CREATE TABLE IF NOT EXISTS provider_profiles (
                    profile_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    config_json TEXT NOT NULL
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

    def _require_client(self) -> BatchClient:
        client = self._client
        if client is None:
            raise BatchTransportError(
                "no batch client is configured; call set_client() before submission or polling"
            )
        return client


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


def _validate_id(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_batch_info(batch: BatchInfo) -> None:
    if not batch.batch_id or not batch.processing_status:
        raise BatchProtocolError("batch response is missing id/status")


def _dump_json(value: object) -> str:
    return canonical_json(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()
