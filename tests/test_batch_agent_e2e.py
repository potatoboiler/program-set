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
from typing import Any

import pytest
from batch_test_types import (
    strands_messages,
)
from batch_test_types import (
    succeeded as make_succeeded,
)
from pydantic import BaseModel
from strands import Agent, Snapshot, tool
from strands.interrupt import InterruptException

from program_set.anthropic_batch import AnthropicBatchClient
from program_set.anthropic_compaction import encode_compaction_signature
from program_set.batch_model import BatchBackedModel
from program_set.batch_result import (
    BatchResultEnvelope,
    CompactionContentBlock,
    NormalizedMessage,
)
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

    def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo:
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


class FakeNonBetaBatches:
    """Minimal non-beta resource retained so the fake satisfies the SDK protocol."""

    def create(self, *, requests: Sequence[Any]) -> object:
        return {}

    def retrieve(self, batch_id: str) -> object:
        return {}

    def results(self, batch_id: str) -> list[object]:
        return []


class FakeNonBetaMessages:
    def __init__(self) -> None:
        self.batches = FakeNonBetaBatches()


class FakeBetaBatches:
    """SDK-shaped beta batch resources; all operations stay offline."""

    def __init__(self) -> None:
        self.submissions: list[list[BatchRequestEnvelope]] = []
        self._batches: dict[str, tuple[str, list[object]]] = {}
        self.beta_calls: list[tuple[str, list[str]]] = []

    def create(
        self,
        *,
        requests: Sequence[BatchRequestEnvelope],
        betas: list[str],
    ) -> object:
        batch_id = f"msgbatch_beta_{len(self.submissions) + 1}"
        self.beta_calls.append(("create", betas))
        self.submissions.append(list(requests))
        self._batches[batch_id] = ("in_progress", [])
        return {"id": batch_id, "processing_status": "in_progress"}

    def retrieve(self, batch_id: str, *, betas: list[str]) -> object:
        self.beta_calls.append(("retrieve", betas))
        status, _ = self._batches[batch_id]
        return {"id": batch_id, "processing_status": status}

    def results(self, batch_id: str, *, betas: list[str]) -> list[object]:
        self.beta_calls.append(("results", betas))
        return self._batches[batch_id][1]

    def complete(self, batch_id: str, results: list[object]) -> None:
        self._batches[batch_id] = ("ended", results)


class FakeBetaMessages:
    def __init__(self) -> None:
        self.batches = FakeBetaBatches()


class FakeBetaNamespace:
    def __init__(self) -> None:
        self.messages = FakeBetaMessages()


class FakeAnthropicSDKClient:
    def __init__(self) -> None:
        self.beta = FakeBetaNamespace()
        self.messages = FakeNonBetaMessages()

    def with_options(self, *, max_retries: int) -> FakeAnthropicSDKClient:
        assert max_retries == 0
        return self


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
        first_params = first_payload[0]["params"]
        assert first_params["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "look up sample"}]}
        ]
        tool_param = _dict_list(first_params["tools"])[0]
        assert tool_param["name"] == "lookup"
        schema = _dict_value(tool_param["input_schema"])
        assert "query" in _dict_value(schema["properties"])

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
        second_payload = _dict_list(client.submissions[1][0]["params"]["messages"])
        assert any(
            block.get("type") == "tool_use"
            for message in second_payload
            for block in _dict_list(message["content"])
        )
        assert any(
            block.get("type") == "tool_result"
            for message in second_payload
            for block in _dict_list(message["content"])
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


def test_threshold_compaction_round_trips_through_restored_tool_cycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = tmp_path / "compaction-batches.sqlite"
    sdk_client = FakeAnthropicSDKClient()
    beta_batches = sdk_client.beta.messages.batches
    beta = "compact-2026-01-12"
    compacted_content = "private-compaction-summary-canary"
    compacted_signature = "private-compaction-signature-canary"
    compaction_block: CompactionContentBlock = {
        "type": "compaction",
        "content": compacted_content,
        "signature": compacted_signature,
        "encrypted_content": "private-compaction-encrypted-canary",
        "tool_changes": [
            {
                "type": "tool_addition",
                "tool": {"type": "tool_reference", "name": "lookup"},
            }
        ],
    }
    context_management = {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 50_000},
            }
        ]
    }
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a deterministic sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    transport = BatchTransport(AnthropicBatchClient(sdk_client, betas=[beta]), database)
    try:
        initial_model = BatchBackedModel(
            transport, agent_id="compaction-agent", model_id="claude-test"
        )
        initial_model.update_config(context_management=context_management)
        initial_agent = Agent(model=initial_model, tools=[lookup])
        deferred = initial_agent("look up sample")
        assert deferred.stop_reason == "interrupt"
        assert deferred.interrupts is not None and len(deferred.interrupts) == 1
        first_interrupt = deferred.interrupts[0]
        first_request_id = initial_model.queued_request_ids[0]
        first_record = transport.get_request(first_request_id)
        first_snapshot = _round_trip_snapshot(
            initial_agent.take_snapshot(
                preset="session",
                app_data={
                    "agent_id": "compaction-agent",
                    "request_id": first_request_id,
                },
            )
        )

        first_batch = transport.flush()
        assert first_batch is not None
        first_params = beta_batches.submissions[0][0]["params"]
        assert first_params["context_management"] == context_management

        beta_batches.complete(
            first_batch.batch_id,
            [
                {
                    "custom_id": first_record.custom_id,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                compaction_block,
                                {
                                    "type": "tool_use",
                                    "id": "toolu_compaction_lookup",
                                    "name": "lookup",
                                    "input": {"query": "sample"},
                                },
                            ],
                            "stop_reason": "tool_use",
                        },
                    },
                }
            ],
        )
        assert transport.poll(first_batch.batch_id).processing_status == "ended"
        assert transport.get_request(first_request_id).result == {
            "type": "succeeded",
            "message": {
                "content": [
                    compaction_block,
                    {
                        "type": "tool_use",
                        "id": "toolu_compaction_lookup",
                        "name": "lookup",
                        "input": {"query": "sample"},
                    },
                ],
                "stop_reason": "tool_use",
            },
        }
        transport.close()

        # A process restart must recreate the client with the explicit beta.
        transport = BatchTransport(
            AnthropicBatchClient(sdk_client, betas=[beta]), database
        )
        resumed_model = BatchBackedModel(
            transport, agent_id="compaction-agent", model_id="claude-test"
        )
        resumed_model.update_config(context_management=context_management)
        resumed_agent = Agent(model=resumed_model, tools=[lookup])
        resumed_agent.load_snapshot(first_snapshot)
        after_tool = resumed_agent(
            [
                {
                    "interruptResponse": {
                        "interruptId": first_interrupt.id,
                        "response": {"request_id": first_request_id, "ready": True},
                    }
                }
            ]
        )
        assert after_tool.stop_reason == "interrupt"
        assert after_tool.interrupts is not None and len(after_tool.interrupts) == 1
        assert tool_calls == ["sample"]

        second_request_id = resumed_model.queued_request_ids[0]
        second_record = transport.get_request(second_request_id)
        second_snapshot = _round_trip_snapshot(
            resumed_agent.take_snapshot(
                preset="session",
                app_data={
                    "agent_id": "compaction-agent",
                    "request_id": second_request_id,
                },
            )
        )
        second_batch = transport.flush()
        assert second_batch is not None
        second_params = beta_batches.submissions[1][0]["params"]
        assert second_params["context_management"] == context_management
        anthropic_messages = _dict_list(second_params["messages"])
        assistant_blocks = [
            block
            for message in anthropic_messages
            if message["role"] == "assistant"
            for block in _dict_list(message["content"])
        ]
        assert assistant_blocks == [
            compaction_block,
            {
                "type": "tool_use",
                "id": "toolu_compaction_lookup",
                "name": "lookup",
                "input": {"query": "sample"},
            },
        ]
        tool_result_blocks = [
            block
            for message in anthropic_messages
            if message["role"] == "user"
            for block in _dict_list(message["content"])
            if block.get("type") == "tool_result"
        ]
        assert tool_result_blocks == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_compaction_lookup",
                "content": [{"type": "text", "text": "sample-result"}],
            }
        ]
        assert transport.get_request(second_request_id).status == "submitted"

        beta_batches.complete(
            second_batch.batch_id,
            [
                {
                    "custom_id": second_record.custom_id,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": "Done."}],
                            "stop_reason": "end_turn",
                        },
                    },
                }
            ],
        )
        transport.poll(second_batch.batch_id)
        transport.close()

        transport = BatchTransport(
            AnthropicBatchClient(sdk_client, betas=[beta]), database
        )
        final_model = BatchBackedModel(
            transport, agent_id="compaction-agent", model_id="claude-test"
        )
        final_model.update_config(context_management=context_management)
        final_agent = Agent(model=final_model, tools=[lookup])
        final_agent.load_snapshot(second_snapshot)
        completed = final_agent(
            [
                {
                    "interruptResponse": {
                        "interruptId": after_tool.interrupts[0].id,
                        "response": {"request_id": second_request_id, "ready": True},
                    }
                }
            ]
        )
        assert completed.stop_reason == "end_turn"
        assert tool_calls == ["sample"]
        assert beta_batches.beta_calls
        assert all(call_betas == [beta] for _, call_betas in beta_batches.beta_calls)

        captured = capsys.readouterr()
        captured_output = captured.out + captured.err + caplog.text
        assert compacted_content not in captured_output
        assert compacted_signature not in captured_output
        assert encode_compaction_signature(compaction_block) not in captured_output
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
                    strands_messages([{"role": "user", "content": "hello"}]),
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


def test_anthropic_context_management_request_kwarg_is_preserved(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "context-management-kwargs.sqlite")
    model = BatchBackedModel(
        transport, agent_id="context-management-kwargs", model_id="claude-test"
    )
    context_management = {"edits": [{"type": "compact_20260112"}]}

    async def defer() -> None:
        with pytest.raises(InterruptException):
            await anext(
                model.stream(
                    strands_messages([{"role": "user", "content": "hello"}]),
                    context_management=context_management,
                )
            )

    try:
        asyncio.run(defer())
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        params = client.submissions[0][0]["params"]
        assert params["context_management"] == context_management
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


def _tool_use_message() -> NormalizedMessage:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_fake_lookup",
                "name": "lookup",
                "input": {"query": "sample"},
            }
        ],
        "stop_reason": "tool_use",
    }


def _text_message(text: str) -> NormalizedMessage:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


def _succeeded(custom_id: str, message: NormalizedMessage) -> BatchResultEnvelope:
    return {
        "custom_id": custom_id,
        "result": make_succeeded(message),
    }


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return [_dict_value(item) for item in value]


def _dict_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("expected a string-keyed object")
    return {key: field for key, field in value.items()}


def _round_trip_snapshot(snapshot: Snapshot) -> Snapshot:
    return Snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
