"""Single-agent proof of deferred Anthropic batch results across restarts.

Everything remote is deterministic and fake. In particular, the production
adapter raises immediately while a request is pending; it never leaves a
coroutine waiting for the batch to finish.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel
from strands import Agent, Snapshot, tool
from strands.interrupt import InterruptException
from strands.types.content import Messages

from program_set.batch_model import BatchBackedModel
from program_set.batch_transport import (
    BatchInfo,
    BatchRequestEnvelope,
    BatchTransport,
)


class FakeAnthropicBatchClient:
    """In-memory fake for Anthropic's documented batch request/result shapes."""

    def __init__(self) -> None:
        self.submissions: list[list[BatchRequestEnvelope]] = []
        self._batches: dict[str, tuple[str, list[object]]] = {}

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> None:
        return None

    def create_batch(
        self, requests: Sequence[BatchRequestEnvelope], *, validated: object
    ) -> BatchInfo:
        del validated
        batch_id = f"msgbatch_fake_{len(self.submissions) + 1}"
        self.submissions.append(list(requests))
        self._batches[batch_id] = ("in_progress", [])
        return BatchInfo(batch_id, "in_progress")

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        status, _ = self._batches[batch_id]
        return BatchInfo(batch_id, status)

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> list[object]:
        del expected_custom_ids
        return self._batches[batch_id][1]

    def complete(self, batch_id: str, results: list[object]) -> None:
        self._batches[batch_id] = ("ended", results)


def test_batch_result_resumes_fresh_agent_and_completes_tool_cycle(
    tmp_path: Path,
) -> None:
    assert version("strands-agents") == "1.61.1"
    database = tmp_path / "batches.sqlite"
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, database)
    agent_id = "agent-session-e2e"
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a deterministic sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    try:
        initial_model = BatchBackedModel(
            transport, agent_id=agent_id, model_id="claude-test"
        )
        initial_agent = Agent(model=initial_model, tools=[lookup])

        deferred = initial_agent("look up sample")
        assert deferred.stop_reason == "interrupt"
        assert deferred.message["role"] == "user"
        assert deferred.interrupts is not None and len(deferred.interrupts) == 1
        first_interrupt = deferred.interrupts[0]
        first_request_id = initial_model.queued_request_ids[0]
        first_record = transport.get_request(first_request_id)
        assert first_record.status == "queued"
        assert first_record.custom_id != first_record.request_id
        assert client.submissions == []  # enqueue does not make a remote call
        assert tool_calls == []

        first_snapshot = _round_trip_snapshot(
            initial_agent.take_snapshot(
                preset="session",
                app_data={"agent_id": agent_id, "request_id": first_request_id},
            )
        )
        assert first_snapshot.app_data == {
            "agent_id": agent_id,
            "request_id": first_request_id,
        }

        first_batch = transport.flush()
        assert first_batch is not None
        assert first_batch == BatchInfo("msgbatch_fake_1", "in_progress")
        first_payload = client.submissions[0]
        assert len(first_payload) == 1
        assert first_payload[0]["custom_id"] == first_record.custom_id
        first_params = cast(dict[str, Any], first_payload[0]["params"])
        assert first_params["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "look up sample"}]}
        ]
        tool_param = cast(list[dict[str, Any]], first_params["tools"])[0]
        assert tool_param["name"] == "lookup"
        assert "query" in tool_param["input_schema"]["properties"]

        client.complete(
            first_batch.batch_id,
            [
                _succeeded(
                    first_record.custom_id,
                    _tool_use_message(),
                )
            ],
        )
        assert transport.poll(first_batch.batch_id).processing_status == "ended"
        persisted_first = transport.get_request(first_request_id)
        assert persisted_first.status == "succeeded"
        assert persisted_first.result_status == "succeeded"
        assert persisted_first.result == {
            "type": "succeeded",
            "message": _tool_use_message(),
        }
        transport.close()

        # Re-open the DB and construct a new provider and Agent, just as after a
        # process restart. The original request is now read from SQLite.
        transport = BatchTransport(client, database)
        resumed_model = BatchBackedModel(
            transport,
            agent_id=first_snapshot.app_data["agent_id"],
            model_id="claude-test",
        )
        resumed_agent = Agent(model=resumed_model, tools=[lookup])
        resumed_agent.load_snapshot(first_snapshot)
        after_tool_use = resumed_agent(
            [
                {
                    "interruptResponse": {
                        "interruptId": first_interrupt.id,
                        "response": {"request_id": first_request_id, "ready": True},
                    }
                }
            ]
        )

        # The fake Anthropic tool_use response goes through the normal Strands
        # tool executor exactly once, then the tool-result request is deferred.
        assert after_tool_use.stop_reason == "interrupt"
        assert after_tool_use.interrupts is not None
        assert len(after_tool_use.interrupts) == 1
        assert tool_calls == ["sample"]
        assert len(resumed_model.queued_request_ids) == 1
        second_request_id = resumed_model.queued_request_ids[0]
        second_record = transport.get_request(second_request_id)
        assert second_record.status == "queued"
        assert len(client.submissions) == 1  # no second create has happened yet

        second_snapshot = _round_trip_snapshot(
            resumed_agent.take_snapshot(
                preset="session",
                app_data={"agent_id": agent_id, "request_id": second_request_id},
            )
        )
        assert second_snapshot.app_data == {
            "agent_id": agent_id,
            "request_id": second_request_id,
        }

        second_batch = transport.flush()
        assert second_batch is not None
        assert second_batch == BatchInfo("msgbatch_fake_2", "in_progress")
        second_payload = cast(
            list[dict[str, Any]],
            client.submissions[1][0]["params"]["messages"],
        )
        assert any(
            block.get("type") == "tool_use"
            for message in second_payload
            for block in message["content"]
        )
        assert any(
            block.get("type") == "tool_result"
            for message in second_payload
            for block in message["content"]
        )

        client.complete(
            second_batch.batch_id,
            [
                _succeeded(
                    second_record.custom_id,
                    _text_message("The lookup returned sample-result."),
                )
            ],
        )
        assert transport.poll(second_batch.batch_id).processing_status == "ended"
        transport.close()

        # Re-open again to prove that the follow-up response, too, is consumed
        # from durable transport state by a fresh Agent/provider pair.
        transport = BatchTransport(client, database)
        final_agent = Agent(
            model=BatchBackedModel(
                transport,
                agent_id=second_snapshot.app_data["agent_id"],
                model_id="claude-test",
            ),
            tools=[lookup],
        )
        final_agent.load_snapshot(second_snapshot)
        completed = final_agent(
            [
                {
                    "interruptResponse": {
                        "interruptId": after_tool_use.interrupts[0].id,
                        "response": {"request_id": second_request_id, "ready": True},
                    }
                }
            ]
        )

        assert completed.stop_reason == "end_turn"
        assert completed.message["content"] == [
            {"text": "The lookup returned sample-result."}
        ]
        history = final_agent.messages
        assert [message["role"] for message in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert any("toolUse" in block for block in history[1]["content"])
        assert any("toolResult" in block for block in history[2]["content"])
        assert history[3]["content"] == [{"text": "The lookup returned sample-result."}]
        assert completed.interrupts is None or completed.interrupts == []
        final_snapshot = final_agent.take_snapshot(preset="session")
        assert final_snapshot.data["interrupt_state"]["activated"] is False
        assert tool_calls == ["sample"]
        assert [batch.batch_id for batch in transport.list_batches()] == [
            "msgbatch_fake_1",
            "msgbatch_fake_2",
        ]
        assert transport.get_request(first_request_id).status == "succeeded"
        assert transport.get_request(second_request_id).status == "succeeded"
    finally:
        transport.close()


def test_identical_requests_from_distinct_agents_get_distinct_ids(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "agents.sqlite")
    try:
        left_model = BatchBackedModel(
            transport, agent_id="agent-left", model_id="claude-test"
        )
        right_model = BatchBackedModel(
            transport, agent_id="agent-right", model_id="claude-test"
        )
        left = Agent(model=left_model)
        right = Agent(model=right_model)

        left_result = left("same request")
        right_result = right("same request")

        assert left_result.stop_reason == "interrupt"
        assert right_result.stop_reason == "interrupt"
        left_request_id = left_model.queued_request_ids[0]
        right_request_id = right_model.queued_request_ids[0]
        assert left_request_id != right_request_id
        assert (
            transport.get_request(left_request_id).custom_id
            != transport.get_request(right_request_id).custom_id
        )
        assert client.submissions == []
    finally:
        transport.close()


def test_anthropic_request_drops_strands_runtime_kwargs(tmp_path: Path) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "runtime-kwargs.sqlite")
    model = BatchBackedModel(
        transport, agent_id="runtime-kwargs", model_id="claude-test"
    )

    async def defer() -> None:
        with pytest.raises(InterruptException):
            await anext(
                model.stream(
                    cast(Messages, [{"role": "user", "content": "hello"}]),
                    model_state={"response_id": "must-not-leak"},
                    dynamic_trailing_blocks=2,
                    max_tokens=32,
                )
            )

    try:
        asyncio.run(defer())
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        params = client.submissions[0][0]["params"]
        assert params["max_tokens"] == 32
        assert "model_state" not in params
        assert "dynamic_trailing_blocks" not in params
    finally:
        transport.close()


def test_structured_output_is_an_async_generator_stub(tmp_path: Path) -> None:
    transport = BatchTransport(FakeAnthropicBatchClient(), tmp_path / "stub.sqlite")
    model = BatchBackedModel(transport, agent_id="stub", model_id="claude-test")
    stream = model.structured_output(BaseModel, [])
    assert inspect.isasyncgen(stream)

    async def consume() -> None:
        with pytest.raises(NotImplementedError, match="structured output"):
            await anext(stream)

    try:
        asyncio.run(consume())
    finally:
        transport.close()


def _tool_use_message() -> dict[str, object]:
    return {
        "id": "msg_fake_tool_use",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_fake_lookup",
                "name": "lookup",
                "input": {"query": "sample"},
            }
        ],
        "model": "claude-test",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def _text_message(text: str) -> dict[str, object]:
    return {
        "id": "msg_fake_final",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-test",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 9},
    }


def _succeeded(custom_id: str, message: dict[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": message},
    }


def _round_trip_snapshot(snapshot: Snapshot) -> Snapshot:
    return Snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
