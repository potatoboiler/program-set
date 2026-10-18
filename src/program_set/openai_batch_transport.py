"""OpenAI Batch API adapter for the existing SQLite batch transport.

The SDK client is injected, so constructing this adapter never creates a client,
reads credentials, or makes a request. It writes one Responses request per JSONL
line, then converts output/error JSONL into the result shape consumed by
``BatchTransport``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from typing import Literal, Protocol, cast

from program_set.batch_transport import (
    BatchInfo,
    BatchProtocolError,
    BatchRequestEnvelope,
    _field,
    _jsonable,
)
from program_set.openai_responses_request import (
    OpenAIResponsesBatchLine,
    OpenAIResponsesRequest,
    encode_reasoning_item,
)

_ENDPOINT: Literal["/v1/responses"] = "/v1/responses"
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}
_BATCH_FAILURE_STATUSES = {"failed", "expired", "cancelled", "canceled"}


class _OpenAIFilesResource(Protocol):
    def create(self, *, file: object, purpose: Literal["batch"]) -> object: ...

    def content(self, file_id: str) -> object: ...


class _OpenAIBatchesResource(Protocol):
    def create(
        self,
        *,
        input_file_id: str,
        endpoint: Literal["/v1/responses"],
        completion_window: Literal["24h"],
    ) -> object: ...

    def retrieve(self, batch_id: str) -> object: ...


class _OpenAIClient(Protocol):
    files: _OpenAIFilesResource
    batches: _OpenAIBatchesResource


class OpenAIBatchClient:
    """Adapt an injected OpenAI Python SDK client to ``BatchTransport``.

    The SDK calls mirror its public synchronous methods:
    ``files.create(file=..., purpose="batch")``, ``batches.create(...)``,
    ``batches.retrieve(batch_id)``, and ``files.content(file_id)``. A batch's
    terminal OpenAI status is normalized to ``ended`` because the shared
    transport persists per-request outcomes at that point.
    """

    def __init__(self, client: object) -> None:
        self._client = cast(_OpenAIClient, client)
        self._retrieved: dict[str, object] = {}

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> object:
        if not requests:
            raise ValueError("an OpenAI batch must contain at least one request")
        lines: list[OpenAIResponsesBatchLine] = []
        for request in requests:
            custom_id = request["custom_id"]
            body = OpenAIResponsesRequest.model_validate(request["params"])
            lines.append(
                OpenAIResponsesBatchLine(
                    custom_id=custom_id,
                    method="POST",
                    url=_ENDPOINT,
                    body=body,
                )
            )
        return lines

    def create_batch(
        self,
        requests: Sequence[BatchRequestEnvelope],
        *,
        validated: object,
    ) -> BatchInfo:
        if not requests:
            raise ValueError("an OpenAI batch must contain at least one request")
        lines = cast(Sequence[OpenAIResponsesBatchLine], validated)
        client = self._without_retries()
        jsonl_lines = [
            _json_line(line.model_dump(mode="json", exclude_none=True))
            for line in lines
        ]

        # The SDK accepts (filename, file, content_type) as FileTypes. JSONL
        # requires one complete request object on each line and a final newline.
        input_file = BytesIO(("\n".join(jsonl_lines) + "\n").encode("utf-8"))
        uploaded_file = client.files.create(
            file=("batch.jsonl", input_file, "application/jsonl"),
            purpose="batch",
        )
        input_file_id = _field(uploaded_file, "id")
        if not isinstance(input_file_id, str) or not input_file_id:
            raise BatchProtocolError("OpenAI file upload response is missing id")

        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint=_ENDPOINT,
            completion_window="24h",
        )
        info = _batch_info(batch)
        self._retrieved[info.batch_id] = batch
        return info

    def _without_retries(self) -> _OpenAIClient:
        with_options = getattr(self._client, "with_options", None)
        if callable(with_options):
            return cast(_OpenAIClient, with_options(max_retries=0))
        return self._client

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        batch = self._client.batches.retrieve(batch_id)
        info = _batch_info(batch)
        self._retrieved[batch_id] = batch
        return info

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> Iterable[object]:
        batch = self._retrieved.get(batch_id)
        if batch is None:
            batch = self._client.batches.retrieve(batch_id)
            info = _batch_info(batch)
            self._retrieved[batch_id] = batch
            if info.batch_id != batch_id:
                raise BatchProtocolError(
                    f"retrieved batch {info.batch_id!r} for requested {batch_id!r}"
                )

        output_file_id = _field(batch, "output_file_id")
        error_file_id = _field(batch, "error_file_id")
        results: list[dict[str, object]] = []
        seen: set[str] = set()
        for file_id in (output_file_id, error_file_id):
            if isinstance(file_id, str) and file_id:
                for line in _read_jsonl(self._client.files.content(file_id), file_id):
                    result = _normalize_result(line)
                    custom_id = result["custom_id"]
                    assert isinstance(custom_id, str)
                    if custom_id in seen:
                        raise BatchProtocolError(
                            f"duplicate OpenAI result for custom_id {custom_id!r}"
                        )
                    seen.add(custom_id)
                    results.append(result)

        batch_status = _field(batch, "status")
        batch_errors = _field(batch, "errors")
        if batch_errors is not None or batch_status in _BATCH_FAILURE_STATUSES:
            error = _batch_failure_error(batch_status, batch_errors)
            results.extend(
                {
                    "custom_id": custom_id,
                    "result": {"type": "errored", "error": error},
                }
                for custom_id in expected_custom_ids
                if custom_id not in seen
            )

        return results


def _batch_info(batch: object) -> BatchInfo:
    batch_id = _field(batch, "id")
    status = _field(batch, "status")
    if not isinstance(batch_id, str) or not batch_id:
        raise BatchProtocolError("OpenAI batch response is missing id")
    if not isinstance(status, str) or not status:
        raise BatchProtocolError("OpenAI batch response is missing status")
    if status in _TERMINAL_STATUSES:
        status = "ended"
    return BatchInfo(batch_id, status)


def _batch_failure_error(status: object, errors: object) -> object:
    if errors is not None:
        normalized = _jsonable(errors)
        if normalized:
            return normalized
    return {
        "code": f"batch_{status if isinstance(status, str) else 'ended'}",
        "message": "OpenAI batch ended without a result line for this request",
    }


def _normalize_result(line: Mapping[str, object]) -> dict[str, object]:
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise BatchProtocolError("OpenAI result line has no valid custom_id")

    response = _as_mapping(line.get("response"))
    error = line.get("error")
    if error is not None or response is None:
        return {
            "custom_id": custom_id,
            "result": {
                "type": "errored",
                "error": _jsonable(error or {"message": "request has no response"}),
            },
        }

    status_code = response.get("status_code")
    body = _as_mapping(response.get("body"))
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        error_value: object = error
        if error_value is None and body is not None:
            error_value = body.get("error")
        if error_value is None:
            error_value = {
                "status_code": status_code,
                "body": _jsonable(response.get("body")),
            }
        return {
            "custom_id": custom_id,
            "result": {"type": "errored", "error": _jsonable(error_value)},
        }
    if body is None:
        raise BatchProtocolError(
            f"successful OpenAI result for {custom_id!r} has no response body"
        )

    if body.get("status") in {"failed", "cancelled"} or body.get("error") is not None:
        return {
            "custom_id": custom_id,
            "result": {
                "type": "errored",
                "error": _jsonable(body.get("error") or {"status": "failed"}),
            },
        }

    message = _responses_message(body, custom_id)
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": message},
    }


def _responses_message(body: Mapping[str, object], custom_id: str) -> dict[str, object]:
    status = body.get("status")
    if status not in {"completed", "incomplete"}:
        raise BatchProtocolError(
            f"unsupported Responses status {status!r} for {custom_id!r}"
        )

    output = body.get("output", [])
    if not isinstance(output, list):
        raise BatchProtocolError(f"Responses output for {custom_id!r} is not a list")
    content: list[dict[str, object]] = []
    saw_text = False
    for raw_item in output:
        item = _as_mapping(raw_item)
        if item is None:
            raise BatchProtocolError(
                f"Responses output for {custom_id!r} has an invalid item"
            )
        item_type = item.get("type")
        if item_type == "message":
            blocks = item.get("content")
            if not isinstance(blocks, list):
                raise BatchProtocolError(
                    f"Responses message for {custom_id!r} has invalid content"
                )
            for raw_block in blocks:
                block = _as_mapping(raw_block)
                if block is None:
                    raise BatchProtocolError(
                        f"Responses message for {custom_id!r} has an invalid content block"
                    )
                block_type = block.get("type")
                if block_type == "output_text":
                    text = block.get("text")
                    if not isinstance(text, str):
                        raise BatchProtocolError(
                            f"Responses output_text for {custom_id!r} is not text"
                        )
                    content.append({"type": "text", "text": text})
                    saw_text = True
                elif block_type == "refusal":
                    refusal = block.get("refusal")
                    if not isinstance(refusal, str):
                        raise BatchProtocolError(
                            f"Responses refusal for {custom_id!r} is not text"
                        )
                    content.append({"type": "text", "text": refusal})
                    saw_text = True
        elif item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise BatchProtocolError(
                    f"Responses function call for {custom_id!r} has invalid call_id/name"
                )
            if not isinstance(arguments, str):
                raise BatchProtocolError(
                    f"Responses function call arguments for {custom_id!r} are not JSON text"
                )
            try:
                input_value = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise BatchProtocolError(
                    f"Responses function call arguments for {custom_id!r} are invalid JSON"
                ) from exc
            if not isinstance(input_value, dict):
                raise BatchProtocolError(
                    f"Responses function call arguments for {custom_id!r} are not an object"
                )
            content.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": input_value,
                }
            )
        elif item_type == "reasoning":
            try:
                signature = encode_reasoning_item(item)
            except ValueError:
                # Do not echo reasoning contents in the error.
                raise BatchProtocolError(
                    f"Responses reasoning item for {custom_id!r} has an "
                    "unsupported shape"
                ) from None
            content.append({"type": "reasoning", "signature": signature})
        else:
            raise BatchProtocolError(
                f"unsupported Responses output item {item_type!r} for {custom_id!r}"
            )

    output_text = body.get("output_text")
    if not saw_text and isinstance(output_text, str) and output_text:
        content.append({"type": "text", "text": output_text})

    has_function_call = any(block["type"] == "tool_use" for block in content)
    incomplete = _as_mapping(body.get("incomplete_details"))
    if has_function_call:
        stop_reason = "tool_use"
    elif incomplete is not None and incomplete.get("reason") == "max_output_tokens":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return {"content": content, "stop_reason": stop_reason}


def _read_jsonl(file_response: object, file_id: str) -> list[dict[str, object]]:
    text = _file_text(file_response, file_id)
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BatchProtocolError(
                f"OpenAI file {file_id!r} has invalid JSON on line {line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise BatchProtocolError(
                f"OpenAI file {file_id!r} line {line_number} is not an object"
            )
        rows.append(cast(dict[str, object], value))
    return rows


def _file_text(response: object, file_id: str) -> str:
    if isinstance(response, bytes):
        return response.decode("utf-8")
    text = _field(response, "text")
    if isinstance(text, str):
        return text
    content = _field(response, "content")
    if isinstance(content, bytes):
        return content.decode("utf-8")
    raise BatchProtocolError(f"OpenAI file content for {file_id!r} is not text")


def _as_mapping(value: object) -> Mapping[str, object] | None:
    dumped = _jsonable(value)
    return cast(Mapping[str, object], dumped) if isinstance(dumped, Mapping) else None


def _json_line(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
