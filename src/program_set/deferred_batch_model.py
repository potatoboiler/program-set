"""Shared lifecycle for provider-specific batch-backed Strands models.

Raising ``InterruptException`` directly from ``Model.stream`` is undocumented
Strands behavior. It is verified only against Strands 1.61.1; callers own batch
submission, polling, and re-entry.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast

from pydantic import BaseModel
from strands.agent.agent_metadata import AgentMetadata
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from program_set.batch_result import NormalizedMessage
from program_set.batch_json import canonical_json
from program_set.batch_transport import BatchTransport, UnknownRequestError


class DeferredBatchModel(Model):
    """Provider-independent request identity, queue, interrupt, and replay flow."""

    def __init__(
        self,
        transport: BatchTransport,
        *,
        agent_id: str,
        config: dict[str, Any],
        custom_id_prefix: str,
        error_prefix: str,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id must be a non-empty stable identifier")
        self.transport = transport
        self.agent_id = agent_id
        self.config = config
        self.custom_id_prefix = custom_id_prefix
        self.error_prefix = error_prefix
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
    ) -> AsyncGenerator[dict[str, Any]]:
        if False:
            yield {}
        raise NotImplementedError("structured output is not supported by batch models")

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        model_state: dict[str, Any] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        dynamic_trailing_blocks: int = 0,
        invocation_state: dict[str, Any] | None = None,
        cancel_signal: threading.Event | None = None,
        agent_metadata: AgentMetadata | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent]:
        # These are Strands runtime state, not provider request fields. Batch
        # requests carry the complete conversation and do not stream cache points.
        del model_state, dynamic_trailing_blocks
        params = self._request_params(
            messages,
            tool_specs,
            system_prompt,
            system_prompt_content,
            tool_choice,
            kwargs,
        )
        digest = hashlib.sha256(
            canonical_json({"agent_id": self.agent_id, "params": params}).encode()
        ).hexdigest()
        request_id = f"agent-request-{digest[:24]}"
        custom_id = f"{self.custom_id_prefix}-{digest[:24]}"

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

        if record.status in {"queued", "submitting", "submit_unknown", "submitted"}:
            # The provider-raised interrupt is an undocumented Strands 1.61.1
            # dependency; callers own flush/poll/re-entry.
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
                f"{self.error_prefix} {request_id!r} completed as {record.status!r}"
            )

        assert record.result is not None
        assert record.result["type"] == "succeeded"
        message = record.result["message"]
        for event in message_events(message):
            yield event

    def _request_params(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt: str | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
        request_kwargs: Mapping[str, Any],
    ) -> dict[str, object]:
        raise NotImplementedError


def message_events(message: NormalizedMessage) -> list[StreamEvent]:
    """Replay normalized text, reasoning, and function-tool response blocks."""

    events: list[StreamEvent] = [{"messageStart": {"role": "assistant"}}]
    for block in message["content"]:
        if block["type"] == "text":
            delta = {"text": block["text"]}
            start: dict[str, object] = {}
        elif block["type"] == "reasoning":
            delta = {"reasoningContent": {"signature": block["signature"]}}
            start = {}
        elif block["type"] == "tool_use":
            delta = {"toolUse": {"input": canonical_json(block["input"])}}
            start = {"toolUse": {"name": block["name"], "toolUseId": block["id"]}}
        else:
            raise AssertionError(
                f"unsupported normalized batch response block: {block!r}"
            )
        events.extend(
            cast(
                list[StreamEvent],
                [
                    {"contentBlockStart": {"start": start}},
                    {"contentBlockDelta": {"delta": delta}},
                    {"contentBlockStop": {}},
                ],
            )
        )
    events.append({"messageStop": {"stopReason": message["stop_reason"]}})
    return events
