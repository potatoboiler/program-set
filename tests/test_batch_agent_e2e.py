"""Agent and multi-agent proofs of deferred Anthropic batch results.

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
from strands.interrupt import Interrupt, InterruptException
from strands.multiagent.graph import Graph, GraphEdge, GraphNode
from strands.multiagent.swarm import Swarm
from strands.session import SnapshotSessionManager
from strands.storage import LocalFileStorage

from program_set.anthropic_batch import AnthropicBatchClient
from program_set.anthropic_compaction import encode_compaction_signature
from program_set.batch_interrupts import ready_batch_interrupt_responses
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
    SubmissionUncertainError,
    UnknownRequestError,
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


def test_graph_parallel_batch_nodes_resume_with_caller_defined_dependencies(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "graph-batches.sqlite")
    storage = LocalFileStorage(str(tmp_path / "graph-snapshots"))
    session_id = "parallel-graph"
    try:

        def build_graph() -> tuple[
            Graph,
            tuple[BatchBackedModel, BatchBackedModel, BatchBackedModel],
        ]:
            left_model = BatchBackedModel(
                transport, agent_id="graph-left", model_id="claude-test"
            )
            right_model = BatchBackedModel(
                transport, agent_id="graph-right", model_id="claude-test"
            )
            joined_model = BatchBackedModel(
                transport, agent_id="graph-joined", model_id="claude-test"
            )
            left = GraphNode("left", Agent(model=left_model))
            right = GraphNode("right", Agent(model=right_model))
            joined = GraphNode(
                "joined", Agent(model=joined_model), dependencies={left, right}
            )
            return (
                Graph(
                    nodes={"left": left, "right": right, "joined": joined},
                    edges={GraphEdge(left, joined), GraphEdge(right, joined)},
                    entry_points={left, right},
                    session_manager=SnapshotSessionManager(
                        session_id, storage=storage, multi_agent_save_latest_on="node"
                    ),
                    id="parallel-graph",
                ),
                (left_model, right_model, joined_model),
            )

        graph, (left_model, right_model, joined_model) = build_graph()
        first = graph("combine independent findings")
        assert first.interrupts is not None and len(first.interrupts) == 2
        assert set(left_model.queued_request_ids + right_model.queued_request_ids) == {
            interrupt.reason["request_id"] for interrupt in first.interrupts
        }
        assert joined_model.queued_request_ids == []
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        assert len(client.submissions[0]) == 2  # one shared batch for parallel nodes
        _complete_batch(
            client,
            transport,
            left_model.queued_request_ids + right_model.queued_request_ids,
            [_text_message("left finding"), _text_message("right finding")],
        )

        # A new Graph definition restores the runtime through the public session
        # manager; the caller supplies only ready responses, not a scheduling plan.
        restored_graph, (_, _, restored_joined_model) = build_graph()
        second = restored_graph(
            ready_batch_interrupt_responses(first.interrupts, transport)
        )
        assert second.interrupts is not None and len(second.interrupts) == 1
        assert len(restored_joined_model.queued_request_ids) == 1
        joined_request_id = restored_joined_model.queued_request_ids[0]
        assert transport.get_request(joined_request_id).status == "queued"
        assert transport.flush() == BatchInfo("msgbatch_fake_2", "in_progress")
        joined_payload = str(client.submissions[1][0]["params"]["messages"])
        assert "left finding" in joined_payload
        assert "right finding" in joined_payload
        _complete_batch(
            client,
            transport,
            [joined_request_id],
            [_text_message("combined finding")],
        )
        final = restored_graph(
            ready_batch_interrupt_responses(second.interrupts, transport)
        )
        assert final.interrupts == []
        assert final.completed_nodes == 3
    finally:
        transport.close()


def test_swarm_handoff_executes_next_agent_then_resumes_its_batch_request(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    database = tmp_path / "swarm-batches.sqlite"
    transport = BatchTransport(client, database)
    storage = LocalFileStorage(str(tmp_path / "swarm-snapshots"))
    try:

        def build_swarm() -> tuple[Swarm, BatchBackedModel, BatchBackedModel]:
            researcher_model = BatchBackedModel(
                transport, agent_id="swarm-researcher", model_id="claude-test"
            )
            writer_model = BatchBackedModel(
                transport, agent_id="swarm-writer", model_id="claude-test"
            )
            researcher = Agent(
                model=researcher_model, name="researcher", description="Research facts"
            )
            writer = Agent(
                model=writer_model, name="writer", description="Write result"
            )
            return (
                Swarm(
                    [researcher, writer],
                    entry_point=researcher,
                    session_manager=SnapshotSessionManager(
                        "handoff-swarm", storage=storage
                    ),
                    id="handoff-swarm",
                ),
                researcher_model,
                writer_model,
            )

        swarm, researcher_model, writer_model = build_swarm()
        first = swarm("write a short answer")
        assert len(first.interrupts) == 1
        researcher_request_id = researcher_model.queued_request_ids[0]
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        handoff_tools = _dict_list(client.submissions[0][0]["params"]["tools"])
        assert any(
            tool_spec["name"] == "handoff_to_agent" for tool_spec in handoff_tools
        )
        _complete_batch(
            client,
            transport,
            [researcher_request_id],
            [
                {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_handoff_writer",
                            "name": "handoff_to_agent",
                            "input": {
                                "agent_name": "writer",
                                "message": "Draft the answer from the findings.",
                                "context": {"finding": "stable sample"},
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                }
            ],
        )

        after_handoff_tool = swarm(
            ready_batch_interrupt_responses(first.interrupts, transport)
        )
        assert len(after_handoff_tool.interrupts) == 1
        assert writer_model.queued_request_ids == []
        researcher_followup = researcher_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_2", "in_progress")
        _complete_batch(
            client,
            transport,
            [researcher_followup],
            [_text_message("Research complete.")],
        )
        handed_off = swarm(
            ready_batch_interrupt_responses(after_handoff_tool.interrupts, transport)
        )
        assert len(handed_off.interrupts) == 1
        assert len(writer_model.queued_request_ids) == 1
        writer_request_id = writer_model.queued_request_ids[0]
        assert transport.flush() == BatchInfo("msgbatch_fake_3", "in_progress")
        writer_payload = str(client.submissions[2][0]["params"]["messages"])
        assert "Draft the answer from the findings." in writer_payload
        assert "stable sample" in writer_payload
        _complete_batch(
            client, transport, [writer_request_id], [_text_message("The answer.")]
        )
        # Rebuild the same Swarm definition; it must restore its paused handoff
        # through SnapshotSessionManager and not invoke the old Swarm again.
        transport.close()
        transport = BatchTransport(client, database)
        restored_swarm, _, _ = build_swarm()
        completed = restored_swarm(
            ready_batch_interrupt_responses(handed_off.interrupts, transport)
        )
        assert completed.interrupts == []
        assert completed.status.value == "completed"
        assert [node.node_id for node in completed.node_history] == [
            "researcher",
            "writer",
        ]
    finally:
        transport.close()


def test_agent_as_tool_child_resumes_from_parent_public_snapshot(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "agent-tool.sqlite")
    try:

        def build_parent() -> tuple[Agent, BatchBackedModel, BatchBackedModel]:
            child_model = BatchBackedModel(
                transport, agent_id="agent-tool-child", model_id="claude-test"
            )
            child = Agent(
                model=child_model,
                name="child",
                description="Answer the request",
            )
            parent_model = BatchBackedModel(
                transport, agent_id="agent-tool-parent", model_id="claude-test"
            )
            return (
                Agent(model=parent_model, tools=[child.as_tool()]),
                parent_model,
                child_model,
            )

        parent, parent_model, child_model = build_parent()
        first = parent("ask the child")
        assert first.stop_reason == "interrupt"
        parent_request_id = parent_model.queued_request_ids[0]
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        _complete_batch(
            client,
            transport,
            [parent_request_id],
            [
                _tool_use_message(
                    "child", "toolu_single_child", {"input": "answer this"}
                )
            ],
        )
        interrupted_child = parent(
            ready_batch_interrupt_responses(first.interrupts or [], transport)
        )
        assert interrupted_child.stop_reason == "interrupt"
        assert interrupted_child.interrupts
        snapshot = _round_trip_snapshot(parent.take_snapshot(preset="session"))
        child_request_id = child_model.queued_request_ids[-1]

        assert transport.flush() == BatchInfo("msgbatch_fake_2", "in_progress")
        child_payload = str(client.submissions[1][0]["params"]["messages"])
        assert "answer this" in child_payload
        _complete_batch(
            client,
            transport,
            [child_request_id],
            [_text_message("child answer")],
        )

        # Rebuild both agent definitions and restore only the parent's public
        # snapshot; its snapshot owns the interrupted ephemeral child turn.
        resumed_parent, resumed_model, resumed_child_model = build_parent()
        resumed_parent.load_snapshot(snapshot)
        resumed = resumed_parent(
            ready_batch_interrupt_responses(
                interrupted_child.interrupts or [], transport
            )
        )
        assert resumed.stop_reason == "interrupt"
        parent_followup_id = resumed_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_3", "in_progress")
        parent_followup_payload = str(client.submissions[2][0]["params"]["messages"])
        assert "child answer" in parent_followup_payload
        _complete_batch(
            client,
            transport,
            [parent_followup_id],
            [_text_message("parent received the child answer")],
        )
        completed = resumed_parent(
            ready_batch_interrupt_responses(resumed.interrupts or [], transport)
        )
        assert completed.stop_reason == "end_turn"

        second = resumed_parent("ask a different independent child question")
        assert second.stop_reason == "interrupt"
        second_parent_request_id = resumed_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_4", "in_progress")
        _complete_batch(
            client,
            transport,
            [second_parent_request_id],
            [
                _tool_use_message(
                    "child",
                    "toolu_single_child_second",
                    {"input": "independent child request"},
                )
            ],
        )
        second = resumed_parent(
            ready_batch_interrupt_responses(second.interrupts or [], transport)
        )
        assert second.stop_reason == "interrupt"
        second_child_request_id = resumed_child_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_5", "in_progress")
        second_child_payload = str(client.submissions[4][0]["params"]["messages"])
        assert "independent child request" in second_child_payload
        assert "answer this" not in second_child_payload
        assert "child answer" not in second_child_payload
        _complete_batch(
            client,
            transport,
            [second_child_request_id],
            [_text_message("independent child answer")],
        )
        second = resumed_parent(
            ready_batch_interrupt_responses(second.interrupts or [], transport)
        )
        assert second.stop_reason == "interrupt"
        second_parent_followup = resumed_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_6", "in_progress")
        second_parent_payload = str(client.submissions[5][0]["params"]["messages"])
        assert "independent child answer" in second_parent_payload
        _complete_batch(
            client,
            transport,
            [second_parent_followup],
            [_text_message("independent interaction completed")],
        )
        assert (
            resumed_parent(
                ready_batch_interrupt_responses(second.interrupts or [], transport)
            ).stop_reason
            == "end_turn"
        )
    finally:
        transport.close()


@pytest.mark.parametrize("reconstruct_at_grandchild_suspend", [False, True])
def test_nested_agent_tools_resume_parent_child_and_grandchild(
    tmp_path: Path, reconstruct_at_grandchild_suspend: bool
) -> None:
    client = FakeAnthropicBatchClient()
    database = tmp_path / "nested-agents.sqlite"
    transport = BatchTransport(client, database)
    try:

        def build_agents() -> tuple[
            Agent, BatchBackedModel, BatchBackedModel, BatchBackedModel
        ]:
            grandchild_model = BatchBackedModel(
                transport, agent_id="nested-grandchild", model_id="claude-test"
            )
            grandchild = Agent(
                model=grandchild_model,
                name="grandchild",
                description="Do the final lookup",
            )
            child_model = BatchBackedModel(
                transport, agent_id="nested-child", model_id="claude-test"
            )
            child = Agent(
                model=child_model,
                name="child",
                description="Delegate to the grandchild",
                tools=[grandchild.as_tool()],
            )
            parent_model = BatchBackedModel(
                transport, agent_id="nested-parent", model_id="claude-test"
            )
            parent = Agent(model=parent_model, tools=[child.as_tool()])
            return parent, parent_model, child_model, grandchild_model

        parent, parent_model, child_model, grandchild_model = build_agents()

        current = parent("complete a nested task")
        assert current.stop_reason == "interrupt"
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        parent_request_id = parent_model.queued_request_ids[0]
        _complete_batch(
            client,
            transport,
            [parent_request_id],
            [
                _tool_use_message(
                    "child", "toolu_parent_child", {"input": "task for child"}
                )
            ],
        )
        current = parent(
            ready_batch_interrupt_responses(current.interrupts or [], transport)
        )
        assert current.stop_reason == "interrupt"

        assert transport.flush() == BatchInfo("msgbatch_fake_2", "in_progress")
        child_payload = str(client.submissions[1][0]["params"]["messages"])
        assert "task for child" in child_payload
        child_request_id = child_model.queued_request_ids[0]
        _complete_batch(
            client,
            transport,
            [child_request_id],
            [
                _tool_use_message(
                    "grandchild",
                    "toolu_child_grandchild",
                    {"input": "task for grandchild"},
                )
            ],
        )
        current = parent(
            ready_batch_interrupt_responses(current.interrupts or [], transport)
        )
        assert current.stop_reason == "interrupt"  # grandchild deferred at depth 3
        snapshot = _round_trip_snapshot(parent.take_snapshot(preset="session"))

        assert transport.flush() == BatchInfo("msgbatch_fake_3", "in_progress")
        grandchild_payload = str(client.submissions[2][0]["params"]["messages"])
        assert "task for grandchild" in grandchild_payload
        grandchild_request_id = grandchild_model.queued_request_ids[0]
        _complete_batch(
            client,
            transport,
            [grandchild_request_id],
            [_text_message("nested lookup complete")],
        )
        if reconstruct_at_grandchild_suspend:
            transport.close()
            transport = BatchTransport(client, database)
            parent, parent_model, child_model, grandchild_model = build_agents()
            parent.load_snapshot(snapshot)

        current = parent(
            ready_batch_interrupt_responses(current.interrupts or [], transport)
        )
        assert current.stop_reason == "interrupt"  # child continues after grandchild

        assert transport.flush() == BatchInfo("msgbatch_fake_4", "in_progress")
        child_followup_payload = str(client.submissions[3][0]["params"]["messages"])
        assert "nested lookup complete" in child_followup_payload
        child_followup_id = child_model.queued_request_ids[-1]
        _complete_batch(
            client,
            transport,
            [child_followup_id],
            [_text_message("child completed")],
        )
        current = parent(
            ready_batch_interrupt_responses(current.interrupts or [], transport)
        )
        assert current.stop_reason == "interrupt"  # parent continues after child tool

        assert transport.flush() == BatchInfo("msgbatch_fake_5", "in_progress")
        parent_followup_payload = str(client.submissions[4][0]["params"]["messages"])
        assert "child completed" in parent_followup_payload
        parent_followup_id = parent_model.queued_request_ids[-1]
        _complete_batch(
            client,
            transport,
            [parent_followup_id],
            [_text_message("all levels completed")],
        )
        current = parent(
            ready_batch_interrupt_responses(current.interrupts or [], transport)
        )
        assert current.stop_reason == "end_turn"
        assert current.message["content"] == [{"text": "all levels completed"}]
    finally:
        transport.close()


def test_preserved_child_owns_durable_history_across_parent_invocations(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "preserved-child.sqlite")
    storage = LocalFileStorage(str(tmp_path / "child-snapshots"))
    try:

        def build_parent() -> tuple[Agent, BatchBackedModel, Agent, BatchBackedModel]:
            child_model = BatchBackedModel(
                transport, agent_id="persistent-child", model_id="claude-test"
            )
            child = Agent(
                model=child_model,
                name="persistent_child",
                description="Remember prior facts",
                session_manager=SnapshotSessionManager(
                    "persistent-child-session", storage=storage
                ),
            )
            parent_model = BatchBackedModel(
                transport, agent_id="persistent-parent", model_id="claude-test"
            )
            parent = Agent(
                model=parent_model,
                tools=[child.as_tool(preserve_context=True)],
            )
            return parent, parent_model, child, child_model

        parent, parent_model, child, child_model = build_parent()
        first = parent("ask child one")
        assert len(first.interrupts or []) == 1
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        parent_request_id = parent_model.queued_request_ids[-1]
        _complete_batch(
            client,
            transport,
            [parent_request_id],
            [
                _tool_use_message(
                    "persistent_child", "toolu_persist_one", {"input": "ask child one"}
                )
            ],
        )
        first = parent(
            ready_batch_interrupt_responses(first.interrupts or [], transport)
        )
        assert first.stop_reason == "interrupt"
        child_request_id = child_model.queued_request_ids[-1]
        assert any(
            "ask child one" in str(message["content"])
            for message in child.messages
            if message["role"] == "user"
        )
        assert transport.flush() == BatchInfo("msgbatch_fake_2", "in_progress")
        _complete_batch(
            client,
            transport,
            [child_request_id],
            [_text_message("remembered fact")],
        )
        first = parent(
            ready_batch_interrupt_responses(first.interrupts or [], transport)
        )
        assert first.stop_reason == "interrupt"  # parent's post-tool model turn
        assert child.messages[-1]["content"] == [{"text": "remembered fact"}]
        assert transport.flush() == BatchInfo("msgbatch_fake_3", "in_progress")
        parent_followup = parent_model.queued_request_ids[-1]
        _complete_batch(
            client,
            transport,
            [parent_followup],
            [_text_message("first interaction done")],
        )
        parent(ready_batch_interrupt_responses(first.interrupts or [], transport))

        # Reconstruct the child with the same definition/session ID. The public
        # session manager restores its history independently of the parent.
        parent, parent_model, child, child_model = build_parent()
        assert child.messages[-2:]
        assert any(
            block.get("text") == "remembered fact"
            for message in child.messages
            for block in message["content"]
            if isinstance(block, dict)
        )
        second = parent("ask child two")
        assert len(second.interrupts or []) == 1
        assert transport.flush() == BatchInfo("msgbatch_fake_4", "in_progress")
        parent_request_id = parent_model.queued_request_ids[-1]
        _complete_batch(
            client,
            transport,
            [parent_request_id],
            [
                _tool_use_message(
                    "persistent_child", "toolu_persist_two", {"input": "ask child two"}
                )
            ],
        )
        second = parent(
            ready_batch_interrupt_responses(second.interrupts or [], transport)
        )
        assert second.stop_reason == "interrupt"
        child_request_id = child_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_5", "in_progress")
        second_payload = _dict_list(client.submissions[-1][0]["params"]["messages"])
        assert any(
            block.get("text") == "remembered fact"
            for message in second_payload
            for block in _dict_list(message["content"])
        )
        assert any(
            block.get("text") == "ask child two"
            for message in second_payload
            for block in _dict_list(message["content"])
        )
        _complete_batch(
            client,
            transport,
            [child_request_id],
            [_text_message("second answer")],
        )
        second = parent(
            ready_batch_interrupt_responses(second.interrupts or [], transport)
        )
        assert second.stop_reason == "interrupt"
        parent_followup = parent_model.queued_request_ids[-1]
        assert transport.flush() == BatchInfo("msgbatch_fake_6", "in_progress")
        _complete_batch(
            client,
            transport,
            [parent_followup],
            [_text_message("second interaction done")],
        )
        assert (
            parent(
                ready_batch_interrupt_responses(second.interrupts or [], transport)
            ).stop_reason
            == "end_turn"
        )
        assert sum(message["role"] == "user" for message in child.messages) == 2
    finally:
        transport.close()


def test_ready_batch_interrupt_responses_validate_and_leave_other_interrupts_alone(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "interrupt-helper.sqlite")
    try:
        model = BatchBackedModel(
            transport, agent_id="interrupt-helper", model_id="claude-test"
        )
        result = Agent(model=model)("helper test")
        assert result.interrupts is not None
        batch_interrupt = result.interrupts[0]
        assert batch_interrupt.reason["source"] == "program-set.deferred-batch-model"
        foreign = Interrupt(
            "approval-id", "approval-required", {"question": "Proceed?"}
        )
        assert (
            ready_batch_interrupt_responses([batch_interrupt, foreign], transport) == []
        )

        request_id = model.queued_request_ids[0]
        custom_id = transport.get_request(request_id).custom_id
        same_named_foreign = [
            Interrupt("foreign-malformed", "batch-response-pending", ["not a reason"]),
            Interrupt(
                "foreign-plausible",
                "batch-response-pending",
                {"request_id": request_id, "custom_id": custom_id},
            ),
        ]
        assert ready_batch_interrupt_responses(same_named_foreign, transport) == []
        with pytest.raises(UnknownRequestError):
            ready_batch_interrupt_responses(
                [
                    Interrupt(
                        "unknown-id",
                        "batch-response-pending",
                        {
                            "source": "program-set.deferred-batch-model",
                            "request_id": "not-registered",
                            "custom_id": "not-registered-custom",
                        },
                    )
                ],
                transport,
            )

        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        _complete_batch(client, transport, [request_id], [_text_message("ready")])
        assert ready_batch_interrupt_responses(
            [batch_interrupt, foreign], transport
        ) == [
            {
                "interruptResponse": {
                    "interruptId": batch_interrupt.id,
                    "response": {"request_id": request_id, "ready": True},
                }
            }
        ]
        namespaced_interrupt_id = f"v1:agent_as_tool:toolu_parent:{batch_interrupt.id}"
        namespaced = Interrupt(
            namespaced_interrupt_id,
            "batch-response-pending",
            batch_interrupt.reason,
        )
        namespaced_response = ready_batch_interrupt_responses([namespaced], transport)
        assert namespaced_response[0]["interruptResponse"]["interruptId"] == (
            namespaced_interrupt_id
        )
        assert ready_batch_interrupt_responses([foreign], transport) == []

        wrong_custom_id = Interrupt(
            batch_interrupt.id,
            "batch-response-pending",
            {
                "request_id": request_id,
                "source": "program-set.deferred-batch-model",
                "custom_id": "not-the-durable-custom-id",
            },
        )
        with pytest.raises(ValueError, match="does not match durable custom_id"):
            ready_batch_interrupt_responses([wrong_custom_id], transport)
    finally:
        transport.close()


def test_ready_batch_interrupt_responses_surface_terminal_failure(
    tmp_path: Path,
) -> None:
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, tmp_path / "failed-interrupt.sqlite")
    try:
        model = BatchBackedModel(
            transport, agent_id="failed-interrupt", model_id="claude-test"
        )
        result = Agent(model=model)("failed request")
        assert result.interrupts is not None
        request_id = model.queued_request_ids[0]
        assert transport.flush() == BatchInfo("msgbatch_fake_1", "in_progress")
        record = transport.get_request(request_id)
        assert record.batch_id is not None
        client.complete(
            record.batch_id,
            [
                {
                    "custom_id": record.custom_id,
                    "result": {
                        "type": "errored",
                        "error": {"code": "mock_failure", "message": "offline"},
                    },
                }
            ],
        )
        transport.poll(record.batch_id)
        with pytest.raises(RuntimeError, match=request_id):
            ready_batch_interrupt_responses(result.interrupts, transport)
    finally:
        transport.close()


def test_ready_batch_interrupt_responses_surface_uncertain_submission(
    tmp_path: Path,
) -> None:
    class FailingCreateClient(FakeAnthropicBatchClient):
        def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo:
            super().create_batch(requests)
            raise TimeoutError("simulated ambiguous create")

    client = FailingCreateClient()
    transport = BatchTransport(client, tmp_path / "uncertain-interrupt.sqlite")
    try:
        model = BatchBackedModel(
            transport, agent_id="uncertain-interrupt", model_id="claude-test"
        )
        result = Agent(model=model)("uncertain request")
        assert result.interrupts is not None
        request_id = model.queued_request_ids[0]

        with pytest.raises(SubmissionUncertainError) as flush_error:
            transport.flush()
        record = transport.get_request(request_id)
        assert record.status == "submit_unknown"
        assert record.submission_id == flush_error.value.submission_id
        assert flush_error.value.request_ids == (request_id,)

        with pytest.raises(SubmissionUncertainError) as helper_error:
            ready_batch_interrupt_responses(result.interrupts, transport)
        assert helper_error.value.submission_id == record.submission_id
        assert helper_error.value.request_ids == (request_id,)
    finally:
        transport.close()


def _tool_use_message(
    name: str = "lookup",
    tool_use_id: str = "toolu_fake_lookup",
    tool_input: dict[str, object] | None = None,
) -> NormalizedMessage:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": tool_input if tool_input is not None else {"query": "sample"},
            }
        ],
        "stop_reason": "tool_use",
    }


def _text_message(text: str) -> NormalizedMessage:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }


def _complete_batch(
    client: FakeAnthropicBatchClient,
    transport: BatchTransport,
    request_ids: list[str],
    messages: list[NormalizedMessage],
) -> None:
    assert len(request_ids) == len(messages)
    records = [transport.get_request(request_id) for request_id in request_ids]
    batch_ids = {record.batch_id for record in records}
    assert len(batch_ids) == 1 and None not in batch_ids
    batch_id = next(iter(batch_ids))
    assert batch_id is not None
    client.complete(
        batch_id,
        [
            _succeeded(record.custom_id, message)
            for record, message in zip(records, messages, strict=True)
        ],
    )
    assert transport.poll(batch_id).processing_status == "ended"


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
