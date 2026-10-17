"""Dependency-only experiment for deferring a Strands provider response.

EXPERIMENT ONLY: ``InterruptException`` is public, but raising it directly from
``Model.stream`` is not a documented provider control-flow contract. In Strands
1.61.0, ``Agent`` catches it around its AgentStreamStage and turns it into an
``AgentResult(stop_reason="interrupt")``. That catch site, provider exception
telemetry, and hook behavior are implementation details that may change. The
tested throw happens before any model stream chunk, so the deferred attempt has
no model-stream usage/latency events; the SDK still fires its ordinary
``AfterInvocationEvent`` with the interrupt result in this version. Neither
behavior is a telemetry contract. Agent snapshots do not persist a provider's
external queue/job; the caller must persist that separately. This module
intentionally uses only a tiny in-memory fake queue and must not be treated as a
production transport or persistence design.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from strands.agent.agent_metadata import AgentMetadata
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec


@dataclass
class _DeferredJob:
    job_id: str
    response: list[StreamEvent] | None = None


class InMemoryDeferredQueue:
    """Small deterministic stand-in for an external deferred-response queue."""

    def __init__(self, followup_response: list[StreamEvent]) -> None:
        self._jobs_by_request: dict[str, _DeferredJob] = {}
        self._jobs_by_id: dict[str, _DeferredJob] = {}
        self._followup_response = followup_response
        self._next_job_number = 1
        self.submissions: list[str] = []

    def response_or_defer(
        self, messages: Messages, tool_specs: list[ToolSpec] | None
    ) -> tuple[str, list[StreamEvent] | None]:
        """Return a completed response, or create/find the pending job without waiting."""
        request_key = _request_key(messages, tool_specs)
        self.submissions.append(request_key)

        if _contains_tool_result(messages):
            return "tool-followup", self._followup_response

        job = self._jobs_by_request.get(request_key)
        if job is None:
            job_id = f"fake-job-{self._next_job_number}"
            self._next_job_number += 1
            job = _DeferredJob(job_id=job_id)
            self._jobs_by_request[request_key] = job
            self._jobs_by_id[job_id] = job

        return job.job_id, job.response

    def complete(self, job_id: str, response: list[StreamEvent]) -> None:
        """Make a stored mocked provider response available for the next stream call."""
        self._jobs_by_id[job_id].response = response

    @property
    def pending_job_ids(self) -> list[str]:
        return [job.job_id for job in self._jobs_by_id.values() if job.response is None]


class DeferredQueueModel(Model):
    """Minimal custom Model provider that defers once, then yields stored events."""

    def __init__(self, queue: InMemoryDeferredQueue) -> None:
        self.queue = queue
        self.config: dict[str, Any] = {}

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
        raise NotImplementedError("structured output is outside this experiment")

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
        job_id, response = self.queue.response_or_defer(messages, tool_specs)
        if response is None:
            # Undocumented provider throw site under test. This is immediate: no
            # sleep, pending coroutine, fabricated model chunk, or tool result.
            raise InterruptException(
                Interrupt(
                    id=job_id,
                    name="provider-response-deferred",
                    reason={"job_id": job_id, "status": "pending"},
                )
            )

        for event in response:
            yield event


def _request_key(messages: Messages, tool_specs: list[ToolSpec] | None) -> str:
    payload = json.dumps(
        {"messages": messages, "tool_specs": tool_specs or []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _contains_tool_result(messages: Messages) -> bool:
    return any("toolResult" in block for message in messages for block in message["content"])
