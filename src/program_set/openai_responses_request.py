"""Pydantic v2 schema for the OpenAI Responses request subset used by batches.

This schema is intentionally narrower than the full Responses API. It makes the
JSONL boundary explicit: only fields declared here can be uploaded to OpenAI,
and unknown framework/runtime keywords are rejected instead of being forwarded
as request fields. Optional fields are excluded from the serialized body so a
normal request contains only what this adapter actually configures.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResponsesTextInput(_StrictModel):
    role: Literal["user", "assistant", "system", "developer"]
    content: str


class ResponsesReasoningSummary(_StrictModel):
    type: Literal["summary_text"]
    text: str


class ResponsesReasoningContent(_StrictModel):
    type: Literal["reasoning_text"]
    text: str


class ResponsesReasoningItem(_StrictModel):
    """One opaque reasoning item replayed before its following output item."""

    type: Literal["reasoning"]
    id: str
    summary: list[ResponsesReasoningSummary]
    content: list[ResponsesReasoningContent] | None = None
    encrypted_content: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


REASONING_SIGNATURE_PREFIX = "openai-responses-reasoning:v1:"


def encode_reasoning_item(item: Mapping[str, object]) -> str:
    """Encode a reasoning item opaquely for Strands message history."""

    try:
        reasoning = ResponsesReasoningItem.model_validate(item)
    except ValidationError:
        raise ValueError("unsupported reasoning item shape") from None
    payload = json.dumps(
        reasoning.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return REASONING_SIGNATURE_PREFIX + base64.urlsafe_b64encode(payload).decode(
        "ascii"
    )


def decode_reasoning_item(signature: str) -> ResponsesReasoningItem:
    """Decode an opaque reasoning item previously produced by this adapter."""

    if not signature.startswith(REASONING_SIGNATURE_PREFIX):
        raise ValueError("unsupported reasoning signature")
    encoded = signature[len(REASONING_SIGNATURE_PREFIX) :]
    try:
        standard = encoded.translate(str.maketrans("-_", "+/"))
        payload = base64.b64decode(standard.encode("ascii"), validate=True)
        value = json.loads(payload.decode("utf-8"))
    except ValueError, UnicodeDecodeError, binascii.Error:
        raise ValueError("invalid reasoning signature") from None
    try:
        return ResponsesReasoningItem.model_validate(value)
    except ValidationError:
        raise ValueError("invalid reasoning item payload") from None


class ResponsesFunctionCall(_StrictModel):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class ResponsesFunctionCallOutput(_StrictModel):
    type: Literal["function_call_output"]
    call_id: str
    output: str


ResponsesInputItem = Annotated[
    ResponsesTextInput
    | ResponsesReasoningItem
    | ResponsesFunctionCall
    | ResponsesFunctionCallOutput,
    Field(union_mode="left_to_right"),
]


class ResponsesFunctionTool(_StrictModel):
    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, Any]


class ResponsesFunctionToolChoice(_StrictModel):
    type: Literal["function"]
    name: str


class OpenAIResponsesRequest(_StrictModel):
    """Supported subset of an OpenAI ``/v1/responses`` request body."""

    model: str
    input: Annotated[list[ResponsesInputItem] | str, Field(min_length=1)]
    instructions: str | None = None
    max_output_tokens: int = Field(ge=1)
    tools: list[ResponsesFunctionTool] | None = None
    tool_choice: (
        Literal["auto", "none", "required"] | ResponsesFunctionToolChoice | None
    ) = None
    parallel_tool_calls: bool | None = None


class OpenAIResponsesBatchLine(_StrictModel):
    """One JSONL line in an OpenAI batch input file."""

    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/responses"]
    body: OpenAIResponsesRequest
