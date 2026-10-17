"""Feasibility test for the experimental provider-side interrupt throw site.

This pins behavior to strands-agents 1.61.0. It uses mocked provider stream
events and an in-memory fake queue only; it makes no network/API calls.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from typing import Any

from strands import Agent, Snapshot, tool
from strands.hooks.events import AfterInvocationEvent

from program_set.provider_interrupt_experiment import DeferredQueueModel, InMemoryDeferredQueue


class _AfterInvocationRecorder:
    def __init__(self) -> None:
        self.stop_reasons: list[str | None] = []

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        registry.add_callback(AfterInvocationEvent, self.record)

    def record(self, event: AfterInvocationEvent) -> None:
        self.stop_reasons.append(event.result.stop_reason if event.result else None)


def _tool_use_response() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"name": "lookup", "toolUseId": "tool-call-1"}}
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": json.dumps({"query": "sample"})}}
            }
        },
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


def _final_response() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "The lookup returned sample-result."}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


def test_provider_interrupt_snapshot_restore_and_stock_tool_cycle() -> None:
    assert version("strands-agents") == "1.61.0"

    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    queue = InMemoryDeferredQueue(followup_response=_final_response())
    recorder = _AfterInvocationRecorder()
    original = Agent(
        model=DeferredQueueModel(queue),
        tools=[lookup],
        hooks=[recorder],
    )

    interrupted = original("look up sample")

    assert interrupted.stop_reason == "interrupt"
    assert interrupted.interrupts is not None
    assert len(interrupted.interrupts) == 1
    assert interrupted.interrupts[0].name == "provider-response-deferred"
    assert tool_calls == []  # no placeholder tool result was fabricated
    assert [message["role"] for message in original.messages] == ["user"]
    assert len(queue.pending_job_ids) == 1

    job_id = queue.pending_job_ids[0]
    snapshot = original.take_snapshot(preset="session", app_data={"queue_job_id": job_id})
    restored_snapshot = Snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored_snapshot.data["interrupt_state"]["activated"] is True
    assert restored_snapshot.app_data == {"queue_job_id": job_id}

    # Simulate the external worker completing the deferred request after the
    # snapshot. The stored tool-use response is delivered verbatim on re-entry.
    queue.complete(job_id, _tool_use_response())
    restored = Agent(
        model=DeferredQueueModel(queue),
        tools=[lookup],
        hooks=[recorder],
    )
    restored.load_snapshot(restored_snapshot)

    completed = restored(
        [
            {
                "interruptResponse": {
                    "interruptId": interrupted.interrupts[0].id,
                    "response": {"job_id": job_id, "acknowledged": True},
                }
            }
        ]
    )

    assert completed.stop_reason == "end_turn"
    assert completed.message["content"] == [{"text": "The lookup returned sample-result."}]
    assert tool_calls == ["sample"]
    assert recorder.stop_reasons == ["interrupt", "end_turn"]
    assert len(queue.submissions) == 3  # defer, stored tool-use response, tool-result follow-up
