"""Deterministic OpenAI-shaped tests; no credentials or HTTP are used."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from strands import Agent, Snapshot, tool
from strands.interrupt import InterruptException
from strands.types.content import Messages

from program_set.batch_transport import BatchInfo, BatchProtocolError, BatchTransport
from program_set.openai_batch_transport import OpenAIBatchClient
from program_set.openai_responses_batch_model import OpenAIBatchBackedModel


class FakeFiles:
    def __init__(self) -> None:
        self.uploads: list[dict[str, str]] = []
        self.contents: dict[str, str] = {}
        self.content_calls: list[str] = []

    def create(self, *, file: object, purpose: str) -> object:
        filename, content_file, content_type = file  # type: ignore[misc]
        file_id = f"input-{len(self.uploads) + 1}"
        content = content_file.read()
        self.uploads.append(
            {
                "filename": filename,
                "content_type": content_type,
                "purpose": purpose,
                "content": content.decode("utf-8"),
            }
        )
        self.contents[file_id] = content.decode("utf-8")
        return {"id": file_id}

    def content(self, file_id: str) -> object:
        self.content_calls.append(file_id)
        return _TextContent(self.contents[file_id])


class FakeBatches:
    def __init__(self, files: FakeFiles) -> None:
        self.files = files
        self.creates: list[dict[str, object]] = []
        self.retrieves: list[str] = []
        self.batches: dict[str, dict[str, object]] = {}

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
    ) -> object:
        batch_id = f"batch-{len(self.creates) + 1}"
        params: dict[str, object] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        self.creates.append(params)
        batch = {
            "id": batch_id,
            "status": "validating",
            **params,
            "output_file_id": None,
            "error_file_id": None,
            "errors": None,
        }
        self.batches[batch_id] = batch
        return batch

    def retrieve(self, batch_id: str) -> object:
        self.retrieves.append(batch_id)
        return self.batches[batch_id]

    def complete(
        self,
        batch_id: str,
        *,
        output: Sequence[dict[str, object]] = (),
        errors: Sequence[dict[str, object]] = (),
        status: str = "completed",
        batch_errors: object = None,
    ) -> None:
        batch = self.batches[batch_id]
        batch["status"] = status
        batch["errors"] = batch_errors
        if output:
            file_id = f"output-{batch_id}"
            self.files.contents[file_id] = _jsonl(output)
            batch["output_file_id"] = file_id
        if errors:
            file_id = f"errors-{batch_id}"
            self.files.contents[file_id] = _jsonl(errors)
            batch["error_file_id"] = file_id


class _TextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches(self.files)
        self.options: list[dict[str, object]] = []

    def with_options(self, **options: Any) -> FakeOpenAIClient:
        self.options.append(options)
        return self


def test_openai_jsonl_and_unordered_output_and_error_lines(
    tmp_path: Path,
) -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / "openai.sqlite"
    )
    try:
        first = transport.enqueue(
            _request("one"), request_id="request-one", custom_id="custom-one"
        )
        second = transport.enqueue(
            _request("two"), request_id="request-two", custom_id="custom-two"
        )

        submitted = transport.flush()

        assert submitted == BatchInfo("batch-1", "validating")
        upload = sdk_client.files.uploads[0]
        assert upload["filename"] == "batch.jsonl"
        assert upload["content_type"] == "application/jsonl"
        assert upload["purpose"] == "batch"
        input_lines = [json.loads(line) for line in str(upload["content"]).splitlines()]
        assert input_lines == [
            {
                "custom_id": "custom-one",
                "method": "POST",
                "url": "/v1/responses",
                "body": _request("one"),
            },
            {
                "custom_id": "custom-two",
                "method": "POST",
                "url": "/v1/responses",
                "body": _request("two"),
            },
        ]
        assert sdk_client.batches.creates == [
            {
                "input_file_id": "input-1",
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            }
        ]
        assert sdk_client.options == [{"max_retries": 0}]

        # File order is independent of request order; failures live in the
        # separate error file and are normalized into transport result records.
        sdk_client.batches.complete(
            "batch-1",
            output=[_output_line("custom-two", _text_response("two-ok"))],
            errors=[
                {
                    "custom_id": "custom-one",
                    "response": None,
                    "error": {"code": "invalid_request_error", "message": "bad"},
                }
            ],
        )
        assert transport.poll("batch-1") == BatchInfo("batch-1", "ended")

        first_result = transport.get_request(first.request_id)
        second_result = transport.get_request(second.request_id)
        assert first_result.status == "errored"
        assert first_result.result == {
            "type": "errored",
            "error": {"code": "invalid_request_error", "message": "bad"},
        }
        assert second_result.status == "succeeded"
        assert second_result.result == {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": "two-ok"}],
                "stop_reason": "end_turn",
            },
        }
        assert sdk_client.batches.retrieves == ["batch-1"]
        assert sdk_client.files.content_calls == ["output-batch-1", "errors-batch-1"]
    finally:
        transport.close()


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled", "canceled"])
def test_all_openai_terminal_statuses_end_transport_batch(
    tmp_path: Path, status: str
) -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / f"{status}.sqlite"
    )
    try:
        enqueued = transport.enqueue(
            _request("terminal"), request_id="terminal-request", custom_id="terminal"
        )
        transport.flush()
        sdk_client.batches.complete(
            "batch-1",
            status=status,
            errors=[
                {
                    "custom_id": "terminal",
                    "response": None,
                    "error": {"data": [{"code": f"batch_{status}"}]},
                }
            ],
        )

        assert transport.poll("batch-1") == BatchInfo("batch-1", "ended")
        record = transport.get_request(enqueued.request_id)
        assert record.status == "errored"
        assert record.result_status == "errored"
        assert record.result == {
            "type": "errored",
            "error": {"data": [{"code": f"batch_{status}"}]},
        }
        assert "input-1" not in sdk_client.files.content_calls
    finally:
        transport.close()


def test_missing_openai_result_is_reported_by_shared_transport(
    tmp_path: Path,
) -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / "missing-result.sqlite"
    )
    try:
        transport.enqueue(
            _request("missing"), request_id="missing-request", custom_id="missing"
        )
        transport.flush()
        sdk_client.batches.complete("batch-1")

        with pytest.raises(BatchProtocolError, match="omitted custom_ids"):
            transport.poll("batch-1")
        assert "input-1" not in sdk_client.files.content_calls
    finally:
        transport.close()


@pytest.mark.parametrize(
    ("batch_errors", "expected_error"),
    [
        (
            {"data": [{"code": "batch_failed", "message": "failed"}]},
            {"data": [{"code": "batch_failed", "message": "failed"}]},
        ),
        (
            None,
            {
                "code": "batch_failed",
                "message": "OpenAI batch ended without a result line for this request",
            },
        ),
    ],
)
def test_failed_batch_recovers_missing_request_results(
    tmp_path: Path, batch_errors: object, expected_error: object
) -> None:
    sdk_client = FakeOpenAIClient()
    database = tmp_path / "batch-errors.sqlite"
    transport = BatchTransport(OpenAIBatchClient(sdk_client), database)
    try:
        first = transport.enqueue(
            _request("first"), request_id="first-request", custom_id="first"
        )
        second = transport.enqueue(
            _request("second"), request_id="second-request", custom_id="second"
        )
        transport.flush()
        sdk_client.batches.complete(
            "batch-1", status="failed", batch_errors=batch_errors
        )
        transport.close()
        transport = BatchTransport(OpenAIBatchClient(sdk_client), database)

        assert transport.poll("batch-1") == BatchInfo("batch-1", "ended")
        for enqueued in (first, second):
            record = transport.get_request(enqueued.request_id)
            assert record.status == "errored"
            assert record.result_status == "errored"
            assert record.result == {"type": "errored", "error": expected_error}
        assert sdk_client.files.content_calls == []
    finally:
        transport.close()


def test_responses_output_text_fallback_and_incomplete_token_limit(
    tmp_path: Path,
) -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / "incomplete.sqlite"
    )
    try:
        enqueued = transport.enqueue(
            _request("incomplete"),
            request_id="incomplete-request",
            custom_id="incomplete",
        )
        transport.flush()
        sdk_client.batches.complete(
            "batch-1",
            output=[
                _output_line(
                    "incomplete",
                    {
                        "id": "resp_incomplete",
                        "object": "response",
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [],
                        "output_text": "partial response",
                    },
                )
            ],
        )

        assert transport.poll("batch-1") == BatchInfo("batch-1", "ended")
        assert transport.get_request(enqueued.request_id).result == {
            "type": "succeeded",
            "message": {
                "content": [{"type": "text", "text": "partial response"}],
                "stop_reason": "max_tokens",
            },
        }
    finally:
        transport.close()


def test_openai_batch_result_resumes_fresh_agent_and_completes_tool_cycle(
    tmp_path: Path,
) -> None:
    assert version("strands-agents") == "1.61.1"
    database = tmp_path / "agent.sqlite"
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(OpenAIBatchClient(sdk_client), database)
    agent_id = "openai-agent-session-e2e"
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a deterministic sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    try:
        initial_model = OpenAIBatchBackedModel(
            transport, agent_id=agent_id, model_id="gpt-4o-mini-test"
        )
        initial_model.update_config(
            instructions="Call lookup exactly once.",
            tool_choice={"type": "function", "name": "lookup"},
            parallel_tool_calls=False,
        )
        initial_agent = Agent(model=initial_model, tools=[lookup])

        deferred = initial_agent("look up sample")
        assert deferred.stop_reason == "interrupt"
        assert deferred.interrupts is not None and len(deferred.interrupts) == 1
        first_interrupt = deferred.interrupts[0]
        first_request_id = initial_model.queued_request_ids[0]
        first_record = transport.get_request(first_request_id)
        assert first_record.status == "queued"
        assert sdk_client.files.uploads == []
        assert sdk_client.batches.creates == []
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
        assert first_batch == BatchInfo("batch-1", "validating")
        first_input = json.loads(sdk_client.files.uploads[0]["content"].splitlines()[0])
        assert first_input["custom_id"] == first_record.custom_id
        assert first_input["url"] == "/v1/responses"
        assert first_input["body"]["model"] == "gpt-4o-mini-test"
        assert first_input["body"]["max_output_tokens"] == 4096
        assert first_input["body"]["instructions"] == "Call lookup exactly once."
        assert first_input["body"]["input"] == [
            {"role": "user", "content": "look up sample"}
        ]
        assert first_input["body"]["tool_choice"] == {
            "type": "function",
            "name": "lookup",
        }
        assert first_input["body"]["parallel_tool_calls"] is False
        assert "model_state" not in first_input["body"]
        openai_tool = first_input["body"]["tools"][0]
        assert openai_tool["type"] == "function"
        assert openai_tool["name"] == "lookup"
        assert "query" in openai_tool["parameters"]["properties"]

        sdk_client.batches.complete(
            first_batch.batch_id,
            output=[_output_line(first_record.custom_id, _tool_response())],
        )
        assert transport.poll(first_batch.batch_id) == BatchInfo("batch-1", "ended")
        assert transport.get_request(first_request_id).result == {
            "type": "succeeded",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_lookup",
                        "name": "lookup",
                        "input": {"query": "sample"},
                    }
                ],
                "stop_reason": "tool_use",
            },
        }
        transport.close()

        # Re-open the DB and construct a new model/Agent as after process restart.
        transport = BatchTransport(OpenAIBatchClient(sdk_client), database)
        resumed_model = OpenAIBatchBackedModel(
            transport,
            agent_id=first_snapshot.app_data["agent_id"],
            model_id="gpt-4o-mini-test",
        )
        resumed_model.update_config(
            instructions="Call lookup exactly once.",
            tool_choice={"type": "function", "name": "lookup"},
            parallel_tool_calls=False,
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

        assert after_tool_use.stop_reason == "interrupt"
        assert after_tool_use.interrupts is not None
        assert len(after_tool_use.interrupts) == 1
        assert tool_calls == ["sample"]
        second_request_id = resumed_model.queued_request_ids[0]
        second_record = transport.get_request(second_request_id)
        assert second_record.status == "queued"
        assert len(sdk_client.batches.creates) == 1

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
        assert second_batch == BatchInfo("batch-2", "validating")
        second_input = json.loads(
            sdk_client.files.uploads[1]["content"].splitlines()[0]
        )
        assert "model_state" not in second_input["body"]
        input_items = second_input["body"]["input"]
        assert any(item.get("type") == "function_call" for item in input_items)
        assert any(item.get("type") == "function_call_output" for item in input_items)

        sdk_client.batches.complete(
            second_batch.batch_id,
            output=[
                _output_line(
                    second_record.custom_id,
                    _text_response("The lookup returned sample-result."),
                )
            ],
        )
        assert transport.poll(second_batch.batch_id) == BatchInfo("batch-2", "ended")
        transport.close()

        transport = BatchTransport(OpenAIBatchClient(sdk_client), database)
        final_model = OpenAIBatchBackedModel(
            transport,
            agent_id=second_snapshot.app_data["agent_id"],
            model_id="gpt-4o-mini-test",
        )
        final_model.update_config(
            instructions="Call lookup exactly once.",
            tool_choice={"type": "function", "name": "lookup"},
            parallel_tool_calls=False,
        )
        final_agent = Agent(model=final_model, tools=[lookup])
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
        assert [message["role"] for message in final_agent.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert any("toolUse" in block for block in final_agent.messages[1]["content"])
        assert any(
            "toolResult" in block for block in final_agent.messages[2]["content"]
        )
        assert final_agent.messages[3]["content"] == [
            {"text": "The lookup returned sample-result."}
        ]
        assert completed.interrupts is None or completed.interrupts == []
        assert tool_calls == ["sample"]
        assert [batch.batch_id for batch in transport.list_batches()] == [
            "batch-1",
            "batch-2",
        ]
    finally:
        transport.close()


def test_reasoning_item_round_trips_opaquely_before_function_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reasoning_item = {
        "id": "rs_fake_reasoning",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "summary canary with spaces"}],
        "content": [
            {"type": "reasoning_text", "text": "chain of thought canary with spaces"}
        ],
        "encrypted_content": "opaque-encrypted-reasoning",
        "status": "completed",
    }

    first_result, second_input, tool_calls = _run_tool_cycle_to_second_request(
        tmp_path, [reasoning_item, _function_call_item()]
    )

    first_message = cast(dict[str, object], first_result["message"])
    first_blocks = cast(list[dict[str, object]], first_message["content"])
    assert [block["type"] for block in first_blocks] == [
        "reasoning",
        "tool_use",
    ]
    # The transport result keeps only the opaque signature, never the text.
    serialized_result = json.dumps(first_result)
    assert "summary canary with spaces" not in serialized_result
    assert "chain of thought canary with spaces" not in serialized_result

    assert second_input[0] == {"role": "user", "content": "look up sample"}
    assert second_input[1] == reasoning_item
    assert [item.get("type") for item in second_input if "type" in item] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert tool_calls == ["sample"]

    captured = capsys.readouterr()
    assert "summary canary with spaces" not in captured.out + captured.err
    assert "chain of thought canary with spaces" not in captured.out + captured.err


def test_tool_cycle_without_reasoning_item_has_no_reasoning_input(
    tmp_path: Path,
) -> None:
    first_result, second_input, tool_calls = _run_tool_cycle_to_second_request(
        tmp_path, [_function_call_item()]
    )

    first_message = cast(dict[str, object], first_result["message"])
    first_blocks = cast(list[dict[str, object]], first_message["content"])
    assert [block["type"] for block in first_blocks] == ["tool_use"]
    assert all(item.get("type") != "reasoning" for item in second_input)
    assert [item.get("type") for item in second_input if "type" in item] == [
        "function_call",
        "function_call_output",
    ]
    assert tool_calls == ["sample"]


def test_opaque_reasoning_signature_round_trips_urlsafe_alphabet() -> None:
    from program_set.openai_responses_request import (
        REASONING_SIGNATURE_PREFIX,
        decode_reasoning_item,
        encode_reasoning_item,
    )

    item = {
        "id": "rs_urlsafe",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "3f>o"}],
        "status": "completed",
    }
    signature = encode_reasoning_item(item)
    encoded = signature[len(REASONING_SIGNATURE_PREFIX) :]
    assert set(encoded) & set("-_")

    decoded = decode_reasoning_item(signature)
    assert decoded.model_dump(mode="json", exclude_none=True) == item


def test_reasoning_input_item_schema_rejects_extra_fields() -> None:
    from program_set.openai_responses_request import OpenAIResponsesRequest

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        OpenAIResponsesRequest.model_validate(
            {
                "model": "gpt-4o-mini-test",
                "max_output_tokens": 16,
                "input": [
                    {
                        "type": "reasoning",
                        "id": "rs_fake_reasoning",
                        "summary": [],
                        "unexpected_reasoning_field": "must-not-leak",
                    }
                ],
            }
        )


@pytest.mark.parametrize("model_state", [{}, {"response_id": "resp_must_not_leak"}])
def test_strands_model_state_never_reaches_responses_jsonl(
    tmp_path: Path, model_state: dict[str, Any]
) -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / "model-state.sqlite"
    )
    try:
        model = OpenAIBatchBackedModel(
            transport, agent_id="model-state", model_id="gpt-4o-mini-test"
        )
        agent = Agent(model=model)
        # Strands forwards agent._model_state to Model.stream on every call.
        agent._model_state = model_state

        result = agent("hello")
        assert result.stop_reason == "interrupt"

        transport.flush()
        input_line = json.loads(sdk_client.files.uploads[0]["content"].splitlines()[0])
        body = input_line["body"]
        assert "model_state" not in body
        assert body == {
            "model": "gpt-4o-mini-test",
            "max_output_tokens": 4096,
            "input": [{"role": "user", "content": "hello"}],
        }
    finally:
        transport.close()


def test_runtime_state_does_not_change_openai_request_identity(
    tmp_path: Path,
) -> None:
    transport = BatchTransport(
        OpenAIBatchClient(FakeOpenAIClient()), tmp_path / "runtime-identity.sqlite"
    )
    model = OpenAIBatchBackedModel(
        transport, agent_id="runtime-identity", model_id="gpt-4o-mini-test"
    )

    async def defer(model_state: dict[str, Any], dynamic_trailing_blocks: int) -> None:
        with pytest.raises(InterruptException):
            await anext(
                model.stream(
                    cast(Messages, [{"role": "user", "content": "hello"}]),
                    model_state=model_state,
                    dynamic_trailing_blocks=dynamic_trailing_blocks,
                )
            )

    try:
        asyncio.run(defer({}, 0))
        request_id = model.queued_request_ids[0]
        asyncio.run(defer({"response_id": "must-not-affect-identity"}, 2))
        assert model.queued_request_ids == [request_id]
        assert transport.get_request(request_id).status == "queued"
    finally:
        transport.close()


def test_unsupported_responses_fields_are_rejected_before_upload(
    tmp_path: Path,
) -> None:
    from program_set.openai_responses_batch_model import _responses_params

    valid = {
        "model": "gpt-4o-mini-test",
        "max_output_tokens": 16,
        "input": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _responses_params(
            cast(Messages, [{"role": "user", "content": "hello"}]),
            None,
            None,
            None,
            None,
            {"model": "gpt-4o-mini-test", "max_output_tokens": 16},
            {"internal_framework_state": {"trace": "must-not-leak"}},
        )

    sdk_client = FakeOpenAIClient()
    client = OpenAIBatchClient(sdk_client)
    transport = BatchTransport(client, tmp_path / "unsupported-fields.sqlite")
    request_id = "unsupported-extra-request"
    try:
        transport.enqueue(
            {
                **valid,
                "internal_framework_state": {"trace": "must-not-leak"},
            },
            request_id=request_id,
            custom_id="unsupported-extra",
        )
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            transport.flush()
        assert transport.get_request(request_id).status == "queued"
        assert sdk_client.files.uploads == []
    finally:
        transport.close()


def test_unsupported_multimodal_and_structured_output_inputs_are_rejected() -> None:
    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(OpenAIBatchClient(sdk_client), ":memory:")
    try:
        model = OpenAIBatchBackedModel(
            transport, agent_id="unsupported", model_id="gpt-4o-mini-test"
        )
        agent = Agent(model=model)
        result = agent("text only")
        assert result.stop_reason == "interrupt"

        # The explicit request mapper is intentionally text-only. These unit
        # assertions avoid invoking the provider for any unsupported content.
        from program_set.openai_responses_batch_model import _responses_params

        with pytest.raises(NotImplementedError, match="structured output"):
            _responses_params(
                cast(
                    Messages,
                    [{"role": "user", "content": [{"image": "not-supported"}]}],
                ),
                None,
                None,
                None,
                None,
                {"model": "gpt-test", "max_output_tokens": 10},
                {"text": {"format": {"type": "json_schema"}}},
            )
        with pytest.raises(NotImplementedError, match="unsupported user content"):
            _responses_params(
                cast(
                    Messages,
                    [{"role": "user", "content": [{"image": "not-supported"}]}],
                ),
                None,
                None,
                None,
                None,
                {"model": "gpt-test", "max_output_tokens": 10},
                {},
            )
    finally:
        transport.close()


def _run_tool_cycle_to_second_request(
    tmp_path: Path, first_output: Sequence[dict[str, object]]
) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
    """Run one batch-backed tool turn and return the queued follow-up input."""

    sdk_client = FakeOpenAIClient()
    transport = BatchTransport(
        OpenAIBatchClient(sdk_client), tmp_path / "reasoning.sqlite"
    )
    agent_id = "openai-reasoning-session"
    tool_calls: list[str] = []

    @tool
    def lookup(query: str) -> str:
        """Look up a deterministic sample value."""
        tool_calls.append(query)
        return f"{query}-result"

    try:
        initial_model = OpenAIBatchBackedModel(
            transport, agent_id=agent_id, model_id="gpt-4o-mini-test"
        )
        initial_model.update_config(
            instructions="Call lookup exactly once.",
            tool_choice={"type": "function", "name": "lookup"},
            parallel_tool_calls=False,
        )
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
                app_data={"agent_id": agent_id, "request_id": first_request_id},
            )
        )

        first_batch = transport.flush()
        assert first_batch is not None
        sdk_client.batches.complete(
            first_batch.batch_id,
            output=[
                _output_line(
                    first_record.custom_id,
                    {
                        "id": "resp_fake_tool",
                        "object": "response",
                        "status": "completed",
                        "output": list(first_output),
                    },
                )
            ],
        )
        assert transport.poll(first_batch.batch_id) == BatchInfo(
            first_batch.batch_id, "ended"
        )
        first_result = transport.get_request(first_request_id)
        assert first_result.result is not None

        restored_model = OpenAIBatchBackedModel(
            transport, agent_id=agent_id, model_id="gpt-4o-mini-test"
        )
        restored_model.update_config(
            instructions="Call lookup exactly once.",
            tool_choice={"type": "function", "name": "lookup"},
            parallel_tool_calls=False,
        )
        restored_agent = Agent(model=restored_model, tools=[lookup])
        restored_agent.load_snapshot(first_snapshot)
        after_tool = restored_agent(
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
        assert tool_calls == ["sample"]

        second_batch = transport.flush()
        assert second_batch is not None
        second_line = json.loads(sdk_client.files.uploads[1]["content"].splitlines()[0])
        return dict(first_result.result), second_line["body"]["input"], tool_calls
    finally:
        transport.close()


def _request(text: str) -> dict[str, object]:
    return {
        "model": "gpt-4o-mini-test",
        "max_output_tokens": 24,
        "input": [{"role": "user", "content": text}],
    }


def _text_response(text: str) -> dict[str, object]:
    return {
        "id": "resp_fake_text",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_fake_text",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _function_call_item() -> dict[str, object]:
    return {
        "id": "fc_fake_lookup",
        "type": "function_call",
        "call_id": "call_lookup",
        "name": "lookup",
        "arguments": '{"query":"sample"}',
    }


def _tool_response() -> dict[str, object]:
    return {
        "id": "resp_fake_tool",
        "object": "response",
        "status": "completed",
        "output": [_function_call_item()],
    }


def _output_line(custom_id: str, completion: dict[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req-{custom_id}",
            "body": completion,
        },
        "error": None,
    }


def _jsonl(rows: Sequence[dict[str, object]]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def _round_trip_snapshot(snapshot: Snapshot) -> Snapshot:
    return Snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
