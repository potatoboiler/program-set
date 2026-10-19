"""Normalized terminal results shared by batch providers and the transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

type StopReason = Literal["end_turn", "tool_use", "max_tokens"]
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


type NormalizedContentBlock = (
    TextContentBlock | ToolUseContentBlock | ReasoningContentBlock
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
    SucceededBatchResult
    | ErroredBatchResult
    | CanceledBatchResult
    | ExpiredBatchResult
)


class BatchResultEnvelope(TypedDict):
    custom_id: str
    result: BatchResult


class BatchResultError(ValueError):
    """A provider or stored result is not in the normalized result format."""


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
    stop_reason = mapping.get("stop_reason")
    if stop_reason == "end_turn" or stop_reason == "stop_sequence":
        # Older SQLite rows stored Anthropic's raw stop reason.
        normalized_stop_reason: StopReason = "end_turn"
    elif stop_reason == "tool_use":
        normalized_stop_reason = "tool_use"
    elif stop_reason == "max_tokens":
        normalized_stop_reason = "max_tokens"
    else:
        raise BatchResultError(f"unsupported normalized stop reason {stop_reason!r}")

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
            raise BatchResultError("normalized tool_use input is not a string-keyed object")
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
    raise BatchResultError(f"unsupported normalized content block type {block_type!r}")


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BatchResultError(f"{description} is not a string-keyed object")
    return value
