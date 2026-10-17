"""Batch-backed Strands model adapter.

This module turns ordinary ``Model.stream`` calls into Anthropic Messages batch
requests and replays a succeeded batch result as normal Strands stream events.
It performs no network I/O itself: the injected
:class:`~program_set.batch_transport.BatchTransport` owns submission, polling,
and persistence.

When a request is still pending, the adapter raises Strands'
``InterruptException`` directly from ``Model.stream``. In Strands 1.61.0 the
agent runtime catches that exception and returns an interrupt result, but raising
it from a provider is an undocumented implementation dependency, not a stable
public ``Model`` contract. Callers are responsible for flushing, polling, and
re-entering the agent; this module intentionally has no retry, lock, or limit
policy.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from pydantic import BaseModel

from strands.agent.agent_metadata import AgentMetadata
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from program_set.batch_transport import (
    BatchTransport,
    UnknownRequestError,
)


class BatchBackedModel(Model):
    """Strands model adapter backed by an application-owned batch transport."""

    def __init__(
        self,
        transport: BatchTransport,
        *,
        agent_id: str,
        model_id: str,
        max_tokens: int = 4096,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty stable identifier")
        if not model_id:
            raise ValueError("model_id must be a non-empty model identifier")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.transport = transport
        self.agent_id = agent_id
        self.config: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
        }
        self.queued_request_ids: list[str] = []

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    async def structured_output(
        self,
        output_model: type[BaseModel],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Keep the return value an async generator while structured output is unsupported.
        if False:
            yield {}
        raise NotImplementedError(
            "structured output is not supported by BatchBackedModel"
        )

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: threading.Event | None = None,
        agent_metadata: AgentMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        params = _anthropic_params(
            messages,
            tool_specs,
            system_prompt,
            system_prompt_content,
            tool_choice,
            self.config,
            kwargs,
        )
        identity = {"agent_id": self.agent_id, "params": params}
        digest = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        request_id = f"agent-request-{digest[:24]}"
        custom_id = f"anthropic-custom-{digest[:24]}"

        try:
            record = self.transport.get_request(request_id)
        except UnknownRequestError:
            self.transport.enqueue(
                params,
                request_id=request_id,
                custom_id=custom_id,
                metadata={"purpose": "strands-model-invocation"},
            )
            self.queued_request_ids.append(request_id)
            record = self.transport.get_request(request_id)

        if record.status in {
            "queued",
            "submitting",
            "submit_unknown",
            "submitted",
        }:
            # The caller owns flush/poll/re-entry. This immediate throw is the
            # experimental Strands 1.61.0 provider interrupt path described in
            # the module docstring, not a documented Model contract.
            raise InterruptException(
                Interrupt(
                    id=request_id,
                    name="batch-response-pending",
                    reason={
                        "request_id": request_id,
                        "custom_id": record.custom_id,
                        "status": record.status,
                    },
                )
            )
        if record.status != "succeeded" or record.result_status != "succeeded":
            raise RuntimeError(
                f"batch request {request_id!r} completed as {record.status!r}"
            )

        assert record.result is not None
        message = record.result["message"]
        assert isinstance(message, Mapping)
        for event in _message_events(message):
            yield event


def _anthropic_params(
    messages: Messages,
    tool_specs: list[ToolSpec] | None,
    system_prompt: str | None,
    system_prompt_content: list[SystemContentBlock] | None,
    tool_choice: ToolChoice | None,
    model_config: Mapping[str, Any],
    extra_params: Mapping[str, Any],
) -> dict[str, object]:
    params: dict[str, object] = dict(model_config)
    params["messages"] = [_anthropic_message(message) for message in messages]
    if system_prompt_content:
        params["system"] = [
            _anthropic_content_block(block) for block in system_prompt_content
        ]
    elif system_prompt:
        params["system"] = system_prompt
    if tool_specs:
        params["tools"] = [
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec["inputSchema"].get(
                    "json", spec["inputSchema"]
                ),
            }
            for spec in tool_specs
        ]
    if tool_choice:
        if "auto" in tool_choice:
            params["tool_choice"] = {"type": "auto"}
        elif "any" in tool_choice:
            params["tool_choice"] = {"type": "any"}
        else:
            params["tool_choice"] = {
                "type": "tool",
                "name": tool_choice["tool"]["name"],
            }
    params.update(extra_params)
    return params


def _anthropic_message(message: Mapping[str, Any]) -> dict[str, object]:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    return {
        "role": message["role"],
        "content": [_anthropic_content_block(block) for block in content],
    }


def _anthropic_content_block(block: Mapping[str, Any]) -> dict[str, object]:
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    if "toolUse" in block:
        tool_use = block["toolUse"]
        return {
            "type": "tool_use",
            "id": tool_use["toolUseId"],
            "name": tool_use["name"],
            "input": tool_use["input"],
        }
    if "toolResult" in block:
        tool_result = block["toolResult"]
        result: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": tool_result["toolUseId"],
            "content": [
                {"type": "text", "text": block["text"]}
                if "text" in block
                else block
                for block in tool_result["content"]
            ],
        }
        if tool_result.get("status") == "error":
            result["is_error"] = True
        return result
    raise AssertionError(f"unsupported Strands content block: {block!r}")


def _message_events(message: Mapping[str, Any]) -> list[StreamEvent]:
    events: list[StreamEvent] = [{"messageStart": {"role": "assistant"}}]
    for block in message["content"]:
        if block["type"] == "text":
            events.extend(
                [
                    {"contentBlockStart": {"start": {}}},
                    {"contentBlockDelta": {"delta": {"text": block["text"]}}},
                    {"contentBlockStop": {}},
                ]
            )
        elif block["type"] == "tool_use":
            events.extend(
                [
                    {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": block["name"],
                                    "toolUseId": block["id"],
                                }
                            }
                        }
                    },
                    {
                        "contentBlockDelta": {
                            "delta": {
                                "toolUse": {
                                    "input": _canonical_json(block["input"])
                                }
                            }
                        }
                    },
                    {"contentBlockStop": {}},
                ]
            )
        else:
            raise AssertionError(f"unsupported Anthropic response block: {block!r}")
    events.append({"messageStop": {"stopReason": message["stop_reason"]}})
    return events


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
