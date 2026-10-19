"""Anthropic Messages Batch API adapter for :mod:`batch_transport`."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, cast

from program_set.batch_json import field, jsonable
from program_set.batch_result import (
    BatchResult,
    BatchResultError,
    NormalizedContentBlock,
    normalize_stop_reason,
    parse_compaction_block,
)
from program_set.batch_transport import (
    BatchInfo,
    BatchProtocolError,
    BatchRequestEnvelope,
)


class _AnthropicBatchesResource(Protocol):
    def create(self, *, requests: Sequence[Any]) -> object: ...

    def retrieve(self, batch_id: str) -> object: ...

    def results(self, batch_id: str) -> Iterable[Any]: ...


class _AnthropicBetaBatchesResource(Protocol):
    def create(self, *, requests: Sequence[Any], betas: list[str]) -> object: ...

    def retrieve(self, batch_id: str, *, betas: list[str]) -> object: ...

    def results(self, batch_id: str, *, betas: list[str]) -> Iterable[Any]: ...


class _AnthropicMessagesResource(Protocol):
    @property
    def batches(self) -> _AnthropicBatchesResource: ...


class _AnthropicBetaMessagesResource(Protocol):
    @property
    def batches(self) -> _AnthropicBetaBatchesResource: ...


class _AnthropicBetaResource(Protocol):
    @property
    def messages(self) -> _AnthropicBetaMessagesResource: ...


class _AnthropicSDKClient(Protocol):
    @property
    def messages(self) -> _AnthropicMessagesResource: ...

    def with_options(self, *, max_retries: int) -> _AnthropicSDKClient: ...


class _AnthropicBetaSDKClient(_AnthropicSDKClient, Protocol):
    @property
    def beta(self) -> _AnthropicBetaResource: ...


class AnthropicBatchClient:
    """Adapt an injected Anthropic SDK client to :class:`BatchClient`.

    The SDK client is injected by the application, so constructing this adapter
    never creates a client, reads credentials, or makes a request. Per-call
    retries are disabled for batch creation so a POST retry after a timeout
    cannot create duplicate remote work.

    ``betas`` explicitly selects Anthropic's beta Messages batch namespace and
    is passed to its batch operations. Recreate this adapter with the same beta
    configuration when reopening a transport after a process restart; beta mode
    is not inferred from persisted request contents.
    """

    def __init__(
        self, client: _AnthropicSDKClient, *, betas: Sequence[str] | None = None
    ) -> None:
        self._client = client
        self._betas = tuple(betas) if betas is not None else None

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> None:
        for request in requests:
            self._validate_request(request)
            if "context_management" in request["params"] and (
                self._betas is None or "compact-2026-01-12" not in self._betas
            ):
                raise ValueError(
                    "Anthropic threshold compaction requires "
                    "AnthropicBatchClient(..., betas=['compact-2026-01-12'])"
                )

    def _validate_request(self, request: BatchRequestEnvelope) -> None:
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
        if "compaction" in params:
            raise ValueError("on-demand Anthropic compaction is not supported")
        if "context_management" in params:
            self._validate_context_management(params["context_management"])

    @staticmethod
    def _validate_context_management(value: object) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("Anthropic context_management must be an object")
        if set(value) != {"edits"}:
            raise ValueError("Anthropic context_management may contain only edits")
        edits = value.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError(
                "Anthropic context_management.edits must be a non-empty list"
            )

        for edit in edits:
            if not isinstance(edit, Mapping):
                raise TypeError("Anthropic context_management edit must be an object")
            if set(edit) - {"type", "trigger"}:
                raise ValueError("Anthropic compaction edit has unsupported fields")
            if edit.get("type") != "compact_20260112":
                raise ValueError(
                    "Anthropic compaction edit type must be compact_20260112"
                )
            if "trigger" not in edit:
                continue

            trigger = edit["trigger"]
            if not isinstance(trigger, Mapping) or set(trigger) != {"type", "value"}:
                raise ValueError(
                    "Anthropic compaction trigger must contain type and value"
                )
            if trigger.get("type") != "input_tokens":
                raise ValueError(
                    "Anthropic compaction trigger type must be input_tokens"
                )
            threshold = trigger.get("value")
            if (
                not isinstance(threshold, int)
                or isinstance(threshold, bool)
                or threshold < 50_000
            ):
                raise ValueError(
                    "Anthropic compaction input_tokens trigger must be an integer "
                    "of at least 50000"
                )

    def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo:
        self.validate_requests(requests)
        client = self._client.with_options(max_retries=0)
        if self._betas is None:
            response = client.messages.batches.create(requests=list(requests))
        else:
            beta_client = cast(_AnthropicBetaSDKClient, client)
            response = beta_client.beta.messages.batches.create(
                requests=list(requests), betas=list(self._betas)
            )
        return self._batch_info(response)

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        if self._betas is None:
            response = self._client.messages.batches.retrieve(batch_id)
        else:
            beta_client = cast(_AnthropicBetaSDKClient, self._client)
            response = beta_client.beta.messages.batches.retrieve(
                batch_id, betas=list(self._betas)
            )
        return self._batch_info(response)

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> Iterable[object]:
        del expected_custom_ids
        if self._betas is None:
            responses = self._client.messages.batches.results(batch_id)
        else:
            beta_client = cast(_AnthropicBetaSDKClient, self._client)
            responses = beta_client.beta.messages.batches.results(
                batch_id, betas=list(self._betas)
            )
        for response in responses:
            custom_id = field(response, "custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise BatchProtocolError(
                    "Anthropic batch result has no valid custom_id"
                )
            yield {
                "custom_id": custom_id,
                "result": self._normalize_result(field(response, "result")),
            }

    @staticmethod
    def _batch_info(value: object) -> BatchInfo:
        batch_id = field(value, "id")
        processing_status = field(value, "processing_status")
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or not isinstance(processing_status, str)
            or not processing_status
        ):
            raise BatchProtocolError("Anthropic batch response is missing id/status")
        return BatchInfo(batch_id, processing_status)

    def _normalize_result(self, value: object) -> BatchResult:
        result_type = field(value, "type")
        if result_type == "succeeded":
            message = field(value, "message")
            content = field(message, "content")
            stop_reason = field(message, "stop_reason")
            if not isinstance(content, list):
                raise BatchProtocolError(
                    "Anthropic result message content is not a list"
                )
            try:
                normalized_stop_reason = normalize_stop_reason(stop_reason)
            except BatchResultError:
                # Fail only this request; other results in the same batch remain
                # eligible for persistence.
                return {
                    "type": "errored",
                    "error": {
                        "code": "unsupported_stop_reason",
                        "message": "Anthropic batch result has an unsupported stop reason",
                    },
                }

            normalized_content = [
                self._normalize_content_block(block) for block in content
            ]
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

    def _normalize_content_block(self, block: object) -> NormalizedContentBlock:
        block_type = field(block, "type")
        if block_type == "text":
            text = field(block, "text")
            if not isinstance(text, str):
                raise BatchProtocolError("Anthropic text block is not text")
            return {"type": "text", "text": text}
        if block_type == "tool_use":
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
            return {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": normalized_input,
            }
        if block_type == "reasoning":
            signature = field(block, "signature")
            if not isinstance(signature, str):
                raise BatchProtocolError("Anthropic reasoning block is malformed")
            return {"type": "reasoning", "signature": signature}
        if block_type == "compaction":
            try:
                return parse_compaction_block(jsonable(block))
            except BatchResultError as exc:
                raise BatchProtocolError(
                    "Anthropic compaction block is malformed"
                ) from exc
        raise BatchProtocolError(
            f"unsupported Anthropic result content block {block_type!r}"
        )
