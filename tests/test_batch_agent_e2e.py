"""Single-agent proof of deferred Anthropic batch results across restarts.

Everything remote is deterministic and fake. In particular, the test adapter
raises immediately while a request is pending; it never leaves a coroutine
waiting for the batch to finish.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from strands import Agent, Snapshot, tool
from strands.agent.agent_metadata import AgentMetadata
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from program_set.batch_transport import (
    BatchInfo,
    BatchTransport,
    MessagesBatchRequest,
    UnknownRequestError,
)


class FakeAnthropicBatchClient:
    """In-memory fake for Anthropic's documented batch request/result shapes."""

    def __init__(self) -> None:
        self.submissions: list[list[MessagesBatchRequest]] = []
        self._batches: dict[str, tuple[str, list[object]]] = {}

    def create_batch(self, requests: Sequence[MessagesBatchRequest]) -> BatchInfo:
        batch_id = f"msgbatch_fake_{len(self.submissions) + 1}"
        self.submissions.append(list(requests))
        self._batches[batch_id] = ("in_progress", [])
        return BatchInfo(batch_id, "in_progress")

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        status, _ = self._batches[batch_id]
        return BatchInfo(batch_id, status)

    def iter_batch_results(self, batch_id: str) -> list[object]:
        return self._batches[batch_id][1]

    def complete(self, batch_id: str, results: list[object]) -> None:
        self._batches[batch_id] = ("ended", results)


class BatchBackedModel(Model):
    """Tiny test adapter between Strands streams and the SQLite batch queue."""

    def __init__(self, transport: BatchTransport) -> None:
        self.transport = transport
        self.config: dict[str, Any] = {}
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
        raise NotImplementedError("structured output is outside this test")

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
        params = _anthropic_params(messages, tool_specs, system_prompt)
        digest = hashlib.sha256(_canonical_json(params).encode()).hexdigest()
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
            # experimental Strands 1.61.0 provider interrupt path, not a
            # documented Model contract.
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


def test_batch_result_resumes_fresh_agent_and_completes_tool_cycle(tmp_path: Path) -> None:
    database = tmp_path / "batches.sqlite"
    client = FakeAnthropicBatchClient()
    transport = BatchTransport(client, database)
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a deterministic sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    try:
        initial_model = BatchBackedModel(transport)
        initial_agent = Agent(model=initial_model, tools=[lookup])

        deferred = initial_agent("look up sample")
        assert deferred.stop_reason == "interrupt"
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
                app_data={"request_id": first_request_id},
            )
        )
        assert first_snapshot.app_data == {"request_id": first_request_id}

        first_batch = transport.flush()
        assert first_batch == BatchInfo("msgbatch_fake_1", "in_progress")
        first_payload = client.submissions[0]
        assert len(first_payload) == 1
        assert first_payload[0]["custom_id"] == first_record.custom_id
        assert first_payload[0]["params"]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "look up sample"}]}
        ]
        tool_param = first_payload[0]["params"]["tools"][0]
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
        resumed_model = BatchBackedModel(transport)
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
                app_data={"request_id": second_request_id},
            )
        )
        assert second_snapshot.app_data == {"request_id": second_request_id}

        second_batch = transport.flush()
        assert second_batch == BatchInfo("msgbatch_fake_2", "in_progress")
        second_payload = client.submissions[1][0]["params"]["messages"]
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
        final_agent = Agent(model=BatchBackedModel(transport), tools=[lookup])
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
        assert tool_calls == ["sample"]
        assert [batch.batch_id for batch in transport.list_batches()] == [
            "msgbatch_fake_1",
            "msgbatch_fake_2",
        ]
        assert transport.get_request(first_request_id).status == "succeeded"
        assert transport.get_request(second_request_id).status == "succeeded"
    finally:
        transport.close()


def _anthropic_params(
    messages: Messages,
    tool_specs: list[ToolSpec] | None,
    system_prompt: str | None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [_anthropic_message(message) for message in messages],
    }
    if system_prompt:
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
