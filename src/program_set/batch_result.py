"""Normalized terminal results shared by batch providers and the transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Required, TypedDict

type StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filtered",
    "limit_total_tokens",
]
type BatchResultType = Literal["succeeded", "errored", "canceled", "expired"]


class TextContentBlock(TypedDict):
    type: Literal["text"]
    text: str


class ToolUseContentBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, object]


class ReasoningContentBlock(TypedDict):
    type: Literal["reasoning"]
    signature: str


class CompactionContentBlock(TypedDict, total=False):
    type: Required[Literal["compaction"]]
    content: str | None
    signature: str | None
    encrypted_content: str | None
    tool_changes: list[dict[str, object]] | None


type NormalizedContentBlock = (
    TextContentBlock
    | ToolUseContentBlock
    | ReasoningContentBlock
    | CompactionContentBlock
)


class NormalizedMessage(TypedDict):
    content: list[NormalizedContentBlock]
    stop_reason: StopReason


class SucceededBatchResult(TypedDict):
    type: Literal["succeeded"]
    message: NormalizedMessage


class ErroredBatchResult(TypedDict):
    type: Literal["errored"]
    error: object


class CanceledBatchResult(TypedDict):
    type: Literal["canceled"]


class ExpiredBatchResult(TypedDict):
    type: Literal["expired"]


type BatchResult = (
    SucceededBatchResult | ErroredBatchResult | CanceledBatchResult | ExpiredBatchResult
)


class BatchResultEnvelope(TypedDict):
    custom_id: str
    result: BatchResult


class BatchResultError(ValueError):
    """A provider or stored result is not in the normalized result format."""


def normalize_stop_reason(value: object) -> StopReason:
    """Map supported Anthropic reasons to Strands stop reasons.

    ``pause_turn`` maps to ``end_turn`` because this deferred response is
    complete for the current invocation. ``refusal`` maps to
    ``content_filtered``, and ``model_context_window_exceeded`` maps to
    ``limit_total_tokens``. ``stop_sequence`` is preserved because Strands
    accepts it as a distinct stop reason.
    """

    if value == "end_turn" or value == "pause_turn":
        return "end_turn"
    if value == "tool_use":
        return "tool_use"
    if value == "max_tokens":
        return "max_tokens"
    if value == "stop_sequence":
        return "stop_sequence"
    if value == "refusal":
        return "content_filtered"
    if value == "content_filtered":
        return "content_filtered"
    if value == "model_context_window_exceeded":
        return "limit_total_tokens"
    if value == "limit_total_tokens":
        return "limit_total_tokens"
    raise BatchResultError(f"unsupported stop reason {value!r}")


def parse_batch_result(value: object) -> BatchResult:
    """Validate an untrusted provider/SQLite result and return its typed form."""

    mapping = _mapping(value, "batch result")
    result_type = mapping.get("type")
    if result_type == "succeeded":
        message = _parse_message(mapping.get("message"))
        return {"type": "succeeded", "message": message}
    if result_type == "errored":
        if "error" not in mapping:
            raise BatchResultError("errored batch result has no error")
        return {"type": "errored", "error": mapping["error"]}
    if result_type == "canceled":
        return {"type": "canceled"}
    if result_type == "expired":
        return {"type": "expired"}
    raise BatchResultError(f"unsupported batch result type {result_type!r}")


def _parse_message(value: object) -> NormalizedMessage:
    mapping = _mapping(value, "succeeded batch result message")
    normalized_stop_reason = normalize_stop_reason(mapping.get("stop_reason"))

    raw_content = mapping.get("content")
    if not isinstance(raw_content, list):
        raise BatchResultError("normalized message content is not a list")
    content = [_parse_content_block(block) for block in raw_content]
    return {"content": content, "stop_reason": normalized_stop_reason}


def _parse_content_block(value: object) -> NormalizedContentBlock:
    mapping = _mapping(value, "normalized message content block")
    block_type = mapping.get("type")
    if block_type == "text":
        text = mapping.get("text")
        if not isinstance(text, str):
            raise BatchResultError("normalized text block has no string text")
        return {"type": "text", "text": text}
    if block_type == "tool_use":
        tool_id = mapping.get("id")
        name = mapping.get("name")
        raw_input = mapping.get("input")
        if not isinstance(tool_id, str) or not isinstance(name, str):
            raise BatchResultError("normalized tool_use block has invalid id/name")
        if not isinstance(raw_input, Mapping):
            raise BatchResultError(
                "normalized tool_use input is not a string-keyed object"
            )
        input_value: dict[str, object] = {}
        for key, item in raw_input.items():
            if not isinstance(key, str):
                raise BatchResultError(
                    "normalized tool_use input is not a string-keyed object"
                )
            input_value[key] = item
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": input_value,
        }
    if block_type == "reasoning":
        signature = mapping.get("signature")
        if not isinstance(signature, str):
            raise BatchResultError("normalized reasoning block has no opaque signature")
        return {"type": "reasoning", "signature": signature}
    if block_type == "compaction":
        return parse_compaction_block(mapping)
    raise BatchResultError(f"unsupported normalized content block type {block_type!r}")


def parse_compaction_block(value: object) -> CompactionContentBlock:
    """Validate a compaction block while preserving optional provider fields."""

    mapping = _mapping(value, "compaction block")
    if mapping.get("type") != "compaction":
        raise BatchResultError("compaction block has an invalid type")

    normalized: CompactionContentBlock = {"type": "compaction"}
    for name in ("content", "signature", "encrypted_content"):
        if name not in mapping:
            continue
        field_value = mapping[name]
        if field_value is not None and not isinstance(field_value, str):
            raise BatchResultError(f"compaction block {name} is not text or null")
        normalized[name] = field_value

    if "tool_changes" in mapping:
        raw_changes = mapping["tool_changes"]
        if raw_changes is not None:
            if not isinstance(raw_changes, list):
                raise BatchResultError("compaction block tool_changes is not a list")
            changes: list[dict[str, object]] = []
            for raw_change in raw_changes:
                change = _mapping(raw_change, "compaction tool change")
                changes.append(dict(change))
            normalized["tool_changes"] = changes
        else:
            normalized["tool_changes"] = None
    return normalized


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BatchResultError(f"{description} is not a string-keyed object")
    return value
