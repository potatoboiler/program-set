from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import anthropic
import httpx2
from batch_test_types import succeeded, successful_result

from program_set.batch_result import BatchResultError, parse_batch_result
from program_set.batch_transport import (
    AnthropicBatchClient,
    BatchInfo,
    BatchProtocolError,
    BatchRequestEnvelope,
    BatchTransport,
    RequestConflictError,
    SubmissionUncertainError,
    _AnthropicSDKClient,
    _normalize_anthropic_result,
)
from program_set.deferred_batch_model import message_events


def request(text: str = "hello") -> dict[str, object]:
    return {
        "model": "claude-test",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": text}],
    }


class FakeBatchClient:
    def __init__(self) -> None:
        self.submissions: list[list[BatchRequestEnvelope]] = []
        self.batch = BatchInfo("msgbatch_test", "in_progress")
        self.results: list[object] = []
        self.fail_create = False

    def validate_requests(self, requests: Sequence[BatchRequestEnvelope]) -> None:
        return None

    def create_batch(self, requests: Sequence[BatchRequestEnvelope]) -> BatchInfo:
        self.submissions.append(list(requests))
        if self.fail_create:
            raise TimeoutError("simulated lost response")
        return self.batch

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        assert batch_id == self.batch.batch_id
        return self.batch

    def iter_batch_results(
        self, batch_id: str, *, expected_custom_ids: Sequence[str]
    ) -> list[object]:
        del expected_custom_ids
        assert batch_id == self.batch.batch_id
        return self.results


class BatchTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Path(self.tempdir.name) / "transport.sqlite"
        self.client = FakeBatchClient()
        self.transport = BatchTransport(self.client, self.database)

    def tearDown(self) -> None:
        self.transport.close()
        self.tempdir.cleanup()

    def test_flush_and_poll_correlate_unordered_results_by_custom_id(self) -> None:
        first = self.transport.enqueue(
            request("first"),
            request_id="work-1",
            custom_id="custom-1",
            metadata={"job": "alpha"},
        )
        second = self.transport.enqueue(
            request("second"),
            request_id="work-2",
            custom_id="custom-2",
            metadata={"job": "beta"},
        )

        submitted = self.transport.flush()
        self.assertEqual(submitted, BatchInfo("msgbatch_test", "in_progress"))
        self.assertEqual(
            [item["custom_id"] for item in self.client.submissions[0]],
            ["custom-1", "custom-2"],
        )
        self.assertEqual(
            self.client.submissions[0][0]["params"]["messages"],
            [{"role": "user", "content": "first"}],
        )
        self.assertEqual(
            self.transport.get_request(first.request_id).status, "submitted"
        )

        self.client.batch = BatchInfo("msgbatch_test", "ended")
        # Deliberately reverse results; the API does not promise request order.
        self.client.results = [
            {
                "custom_id": "custom-2",
                "result": {
                    "type": "errored",
                    "error": {"type": "invalid_request_error", "message": "bad"},
                },
            },
            {
                "custom_id": "custom-1",
                "result": succeeded(
                    {
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                    }
                ),
            },
        ]
        self.assertEqual(
            self.transport.poll("msgbatch_test").processing_status, "ended"
        )

        first_record = self.transport.get_request(first.request_id)
        second_record = self.transport.get_request(second.request_id)
        self.assertEqual(first_record.status, "succeeded")
        self.assertEqual(first_record.result_status, "succeeded")
        message = successful_result(first_record.result)["message"]
        self.assertEqual(message["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(first_record.metadata, {"job": "alpha"})
        self.assertEqual(second_record.status, "errored")
        self.assertEqual(second_record.result_status, "errored")

    def test_batch_registry_and_results_survive_reopening_database(self) -> None:
        self.transport.enqueue(request(), request_id="durable", metadata={"item": "42"})
        self.transport.flush()
        self.client.batch = BatchInfo("msgbatch_test", "ended")
        self.client.results = [{"custom_id": "durable", "result": {"type": "canceled"}}]
        self.transport.poll("msgbatch_test")
        self.transport.close()

        reopened = BatchTransport(self.client, self.database)
        try:
            self.assertEqual(
                reopened.list_batches(), (BatchInfo("msgbatch_test", "ended"),)
            )
            result = reopened.get_request("durable")
            self.assertEqual(result.status, "canceled")
            self.assertEqual(result.result, {"type": "canceled"})
            self.assertEqual(result.metadata, {"item": "42"})
        finally:
            reopened.close()

    def test_ambiguous_submit_is_not_retried_without_explicit_requeue(self) -> None:
        enqueued = self.transport.enqueue(request(), request_id="maybe-sent")
        self.client.fail_create = True

        with self.assertRaises(SubmissionUncertainError) as raised:
            self.transport.flush()
        error = raised.exception
        self.assertEqual(error.request_ids, (enqueued.request_id,))
        self.assertEqual(
            self.transport.get_request(enqueued.request_id).status, "submit_unknown"
        )
        self.assertIsNone(self.transport.flush())
        self.assertEqual(len(self.client.submissions), 1)

        self.client.fail_create = False
        self.assertEqual(self.transport.requeue_uncertain(error.submission_id), 1)
        self.assertEqual(
            self.transport.get_request(enqueued.request_id).status, "queued"
        )
        self.transport.flush()
        self.assertEqual(len(self.client.submissions), 2)

    def test_duplicate_request_id_is_idempotent_but_conflicts_are_rejected(
        self,
    ) -> None:
        enqueued = self.transport.enqueue(
            request(), request_id="same", metadata={"source": "one"}
        )
        self.assertEqual(
            self.transport.enqueue(
                request(), request_id="same", metadata={"source": "one"}
            ),
            enqueued,
        )
        with self.assertRaises(RequestConflictError):
            self.transport.enqueue(
                request("different"), request_id="same", metadata={"source": "one"}
            )

    def test_envelope_ids_and_opaque_serializable_params(self) -> None:
        for id_name in ("request_id", "custom_id"):
            with self.subTest(id_name=id_name), self.assertRaises(ValueError):
                if id_name == "request_id":
                    self.transport.enqueue(
                        {"provider-specific": {"opaque": True}},
                        request_id=" ",
                    )
                else:
                    self.transport.enqueue(
                        {"provider-specific": {"opaque": True}},
                        custom_id=" ",
                    )

        accepted = self.transport.enqueue(
            {"provider-specific": {"stream": True, "speed": "fast"}},
            request_id="opaque-request",
            custom_id="provider.id/opaque",
        )
        self.assertEqual(accepted.custom_id, "provider.id/opaque")

        with self.assertRaises(TypeError):
            self.transport.enqueue(
                ["not a mapping"],
                request_id="not-mapping",
            )
        with self.assertRaises(TypeError):
            self.transport.enqueue(
                {"not-json": object()}, request_id="not-serializable"
            )

    def test_anthropic_validation_fails_before_submission(self) -> None:
        class Batches:
            def __init__(self) -> None:
                self.sent: list[Sequence[BatchRequestEnvelope]] = []

            def create(self, *, requests: Sequence[BatchRequestEnvelope]) -> object:
                self.sent.append(requests)
                return {"id": "remote-id", "processing_status": "in_progress"}

            def retrieve(self, batch_id: str) -> object:
                return {"id": batch_id, "processing_status": "ended"}

            def results(self, batch_id: str) -> list[object]:
                return []

        class Messages:
            def __init__(self) -> None:
                self.batches = Batches()

        class SDKClient:
            def __init__(self) -> None:
                self.messages = Messages()

            def with_options(self, *, max_retries: int) -> SDKClient:
                assert max_retries == 0
                return self

        cases = (
            ("bad-payload", {**request(), "stream": True}, "valid-id"),
            ("bad-id", request(), "contains.dot"),
        )
        for request_id, params, custom_id in cases:
            with self.subTest(request_id=request_id):
                sdk_client = SDKClient()
                transport = BatchTransport(AnthropicBatchClient(sdk_client), ":memory:")
                try:
                    transport.enqueue(
                        params, request_id=request_id, custom_id=custom_id
                    )
                    with self.assertRaises(ValueError):
                        transport.flush()
                    self.assertEqual(transport.get_request(request_id).status, "queued")
                    self.assertEqual(sdk_client.messages.batches.sent, [])
                finally:
                    transport.close()

    def test_real_sdk_batch_create_uses_zero_retries(self) -> None:
        attempts = 0

        def fail_request(_: httpx2.Request) -> httpx2.Response:
            nonlocal attempts
            attempts += 1
            return httpx2.Response(
                500,
                json={"error": {"type": "api_error", "message": "mock failure"}},
            )

        http_client = httpx2.Client(transport=httpx2.MockTransport(fail_request))
        sdk_client = anthropic.Anthropic(api_key="test-only", http_client=http_client)
        try:
            adapter = AnthropicBatchClient(cast(_AnthropicSDKClient, sdk_client))
            requests: list[BatchRequestEnvelope] = [
                {"custom_id": "sdk-test", "params": request()}
            ]
            adapter.validate_requests(requests)
            with self.assertRaises(anthropic.APIStatusError):
                adapter.create_batch(requests)
            self.assertEqual(attempts, 1)
        finally:
            sdk_client.close()

    def test_result_with_unknown_custom_id_is_rejected(self) -> None:
        self.transport.enqueue(request(), request_id="known")
        self.transport.flush()
        self.client.batch = BatchInfo("msgbatch_test", "ended")
        self.client.results = [{"custom_id": "other", "result": {"type": "canceled"}}]
        with self.assertRaises(BatchProtocolError):
            self.transport.poll("msgbatch_test")

    def test_normalized_result_parser_validates_and_discards_provider_extras(
        self,
    ) -> None:
        result = parse_batch_result(
            {
                "type": "succeeded",
                "message": {
                    "id": "provider-message-id",
                    "usage": {"input_tokens": 3},
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "stop_sequence",
                },
            }
        )
        self.assertEqual(
            result,
            {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "stop_sequence",
                },
            },
        )
        with self.assertRaises(BatchResultError):
            parse_batch_result(
                {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": 42}],
                        "stop_reason": "end_turn",
                    },
                }
            )

    def test_anthropic_stop_reasons_normalize_and_replay_as_strands_reasons(
        self,
    ) -> None:
        cases = (
            ("end_turn", "end_turn"),
            ("max_tokens", "max_tokens"),
            ("stop_sequence", "stop_sequence"),
            ("tool_use", "tool_use"),
            ("pause_turn", "end_turn"),
            ("refusal", "content_filtered"),
            ("model_context_window_exceeded", "limit_total_tokens"),
        )
        for provider_reason, expected_reason in cases:
            with self.subTest(provider_reason=provider_reason):
                result = parse_batch_result(
                    {
                        "type": "succeeded",
                        "message": {
                            "content": [],
                            "stop_reason": provider_reason,
                        },
                    }
                )
                assert result["type"] == "succeeded"
                self.assertEqual(result["message"]["stop_reason"], expected_reason)
                self.assertEqual(
                    message_events(result["message"])[-1],
                    {"messageStop": {"stopReason": expected_reason}},
                )

    def test_legacy_sqlite_row_preserves_stop_sequence(self) -> None:
        enqueued = self.transport.enqueue(request(), request_id="legacy-stop-sequence")
        legacy_result = {
            "type": "succeeded",
            "message": {
                "id": "msg_legacy",
                "model": "claude-legacy",
                "usage": {"input_tokens": 2, "output_tokens": 1},
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "stop_sequence",
            },
        }
        with self.transport._connection:
            self.transport._connection.execute(
                "UPDATE requests SET status = 'succeeded', result_status = ?, "
                "result_json = ? WHERE request_id = ?",
                ("succeeded", json.dumps(legacy_result), enqueued.request_id),
            )

        record = self.transport.get_request(enqueued.request_id)

        self.assertEqual(record.result_status, "succeeded")
        self.assertEqual(
            record.result,
            {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "stop_sequence",
                },
            },
        )

    def test_anthropic_nonstandard_stop_reasons_persist_per_request(self) -> None:
        raw_reasons = {
            "refusal": "content_filtered",
            "pause_turn": "end_turn",
            "model_context_window_exceeded": "limit_total_tokens",
            "stop_sequence": "stop_sequence",
            "unknown_reason": None,
        }

        class Batches:
            def __init__(self) -> None:
                self.status = "in_progress"
                self.raw_results: list[object] = []

            def create(self, *, requests: Sequence[BatchRequestEnvelope]) -> object:
                return {"id": "reason-batch", "processing_status": self.status}

            def retrieve(self, batch_id: str) -> object:
                return {"id": batch_id, "processing_status": self.status}

            def results(self, batch_id: str) -> list[object]:
                return self.raw_results

        class Messages:
            def __init__(self) -> None:
                self.batches = Batches()

        class SDKClient:
            def __init__(self) -> None:
                self.messages = Messages()

            def with_options(self, *, max_retries: int) -> SDKClient:
                assert max_retries == 0
                return self

        sdk_client = SDKClient()
        batches = sdk_client.messages.batches
        for provider_reason in raw_reasons:
            batches.raw_results.append(
                {
                    "custom_id": provider_reason,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": provider_reason}],
                            "stop_reason": provider_reason,
                        },
                    },
                }
            )
        transport = BatchTransport(
            AnthropicBatchClient(sdk_client), self.database.with_name("reasons.sqlite")
        )
        try:
            for provider_reason in raw_reasons:
                transport.enqueue(
                    request(provider_reason),
                    request_id=f"request-{provider_reason}",
                    custom_id=provider_reason,
                )
            transport.flush()
            batches.status = "ended"

            transport.poll("reason-batch")

            for provider_reason, expected_reason in raw_reasons.items():
                record = transport.get_request(f"request-{provider_reason}")
                if expected_reason is None:
                    self.assertEqual(record.status, "errored")
                    self.assertEqual(record.result_status, "errored")
                    self.assertEqual(
                        record.result,
                        {
                            "type": "errored",
                            "error": {
                                "code": "unsupported_stop_reason",
                                "message": (
                                    "Anthropic batch result has an unsupported stop reason"
                                ),
                            },
                        },
                    )
                    continue
                self.assertEqual(record.status, "succeeded")
                result = successful_result(record.result)
                self.assertEqual(result["message"]["stop_reason"], expected_reason)
        finally:
            transport.close()

    def test_restart_marks_interrupted_submissions_uncertain(self) -> None:
        enqueued = self.transport.enqueue(request(), request_id="interrupted")
        with self.transport._connection:
            self.transport._connection.execute(
                "UPDATE requests SET status = 'submitting', submission_id = ? "
                "WHERE request_id = ?",
                ("recovered-submission", enqueued.request_id),
            )
        self.transport.close()

        reopened = BatchTransport(self.client, self.database)
        try:
            self.assertEqual(
                reopened.get_request(enqueued.request_id).status, "submit_unknown"
            )
            self.assertIsNone(reopened.flush())
            self.assertEqual(reopened.requeue_uncertain("recovered-submission"), 1)
        finally:
            reopened.close()

    def test_anthropic_adapter_calls_sdk_batch_resources(self) -> None:
        class Batches:
            def __init__(self) -> None:
                self.sent: list[BatchRequestEnvelope] = []

            def create(self, *, requests: Sequence[BatchRequestEnvelope]) -> object:
                self.sent = list(requests)
                return {"id": "remote-id", "processing_status": "in_progress"}

            def retrieve(self, batch_id: str) -> object:
                return {"id": batch_id, "processing_status": "ended"}

            def results(self, batch_id: str) -> list[object]:
                return [
                    {
                        "custom_id": "id",
                        "result": {
                            "type": "succeeded",
                            "message": {
                                "id": "provider-id",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "toolu_lookup",
                                        "name": "lookup",
                                        "input": {"query": "sample"},
                                    }
                                ],
                                "stop_reason": "tool_use",
                                "usage": {"input_tokens": 3},
                            },
                        },
                    }
                ]

        class Messages:
            def __init__(self) -> None:
                self.batches = Batches()

        class SDKClient:
            def __init__(self) -> None:
                self.messages = Messages()

            def with_options(self, *, max_retries: int) -> SDKClient:
                assert max_retries == 0
                return self

        sdk_client = SDKClient()
        adapter = AnthropicBatchClient(sdk_client)
        payload: BatchRequestEnvelope = {"custom_id": "id", "params": request()}
        adapter.validate_requests([payload])
        self.assertEqual(
            adapter.create_batch([payload]).batch_id,
            "remote-id",
        )
        self.assertEqual(sdk_client.messages.batches.sent, [payload])
        self.assertEqual(adapter.retrieve_batch("remote-id").processing_status, "ended")
        self.assertEqual(
            list(adapter.iter_batch_results("remote-id", expected_custom_ids=("id",))),
            [
                {
                    "custom_id": "id",
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "toolu_lookup",
                                    "name": "lookup",
                                    "input": {"query": "sample"},
                                }
                            ],
                            "stop_reason": "tool_use",
                        },
                    },
                }
            ],
        )

    def test_anthropic_beta_adapter_uses_beta_batch_resources(self) -> None:
        class BetaBatches:
            def __init__(self) -> None:
                self.calls: list[tuple[str, list[str]]] = []

            def create(
                self,
                *,
                requests: Sequence[BatchRequestEnvelope],
                betas: list[str],
            ) -> object:
                assert len(requests) == 1
                self.calls.append(("create", betas))
                return {"id": "beta-batch", "processing_status": "in_progress"}

            def retrieve(self, batch_id: str, *, betas: list[str]) -> object:
                assert batch_id == "beta-batch"
                self.calls.append(("retrieve", betas))
                return {"id": batch_id, "processing_status": "ended"}

            def results(self, batch_id: str, *, betas: list[str]) -> list[object]:
                assert batch_id == "beta-batch"
                self.calls.append(("results", betas))
                return [{"custom_id": "beta-id", "result": {"type": "canceled"}}]

        class BetaMessages:
            def __init__(self) -> None:
                self.batches = BetaBatches()

        class Beta:
            def __init__(self) -> None:
                self.messages = BetaMessages()

        class SDKClient:
            def __init__(self) -> None:
                self.beta = Beta()
                self.retry_limits: list[int] = []

            @property
            def messages(self) -> object:
                raise AssertionError("beta requests must use the beta namespace")

            def with_options(self, *, max_retries: int) -> SDKClient:
                self.retry_limits.append(max_retries)
                return self

        sdk_client = SDKClient()
        adapter = AnthropicBatchClient(
            cast(_AnthropicSDKClient, sdk_client),
            betas=["compact-2026-01-12"],
        )
        payload: BatchRequestEnvelope = {
            "custom_id": "beta-id",
            "params": {
                **request(),
                "context_management": {
                    "edits": [
                        {
                            "type": "compact_20260112",
                            "trigger": {"type": "input_tokens", "value": 50_000},
                        }
                    ]
                },
            },
        }

        self.assertEqual(adapter.create_batch([payload]).batch_id, "beta-batch")
        self.assertEqual(
            adapter.retrieve_batch("beta-batch").processing_status, "ended"
        )
        self.assertEqual(
            list(
                adapter.iter_batch_results(
                    "beta-batch", expected_custom_ids=("beta-id",)
                )
            ),
            [{"custom_id": "beta-id", "result": {"type": "canceled"}}],
        )
        self.assertEqual(sdk_client.retry_limits, [0])
        self.assertEqual(
            sdk_client.beta.messages.batches.calls,
            [
                ("create", ["compact-2026-01-12"]),
                ("retrieve", ["compact-2026-01-12"]),
                ("results", ["compact-2026-01-12"]),
            ],
        )

    def test_anthropic_threshold_context_management_validation(self) -> None:
        adapter = AnthropicBatchClient(
            cast(_AnthropicSDKClient, object()), betas=["compact-2026-01-12"]
        )

        def validate(context_management: object) -> None:
            adapter.validate_requests(
                [
                    {
                        "custom_id": "context-test",
                        "params": {
                            **request(),
                            "context_management": context_management,
                        },
                    }
                ]
            )

        validate({"edits": [{"type": "compact_20260112"}]})
        validate(
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 50_000},
                    }
                ]
            }
        )

        invalid_contexts = (
            {"edits": [{"type": "summarize"}]},
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens"},
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "tokens", "value": 50_000},
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 49_999},
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 50_000.0},
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": True},
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": 50_000,
                            "extra": 1,
                        },
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "instructions": "not supported",
                    }
                ]
            },
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "pause_after_compaction": True,
                    }
                ]
            },
            {"edits": [{"type": "compact_20260112"}], "unknown": True},
            {"edits": []},
            {"edits": "not a list"},
            {"edits": [{"type": "compact_20260112"}, "not an edit object"]},
        )
        for context_management in invalid_contexts:
            with (
                self.subTest(context_management=context_management),
                self.assertRaises((TypeError, ValueError)),
            ):
                validate(context_management)

        non_beta_adapter = AnthropicBatchClient(cast(_AnthropicSDKClient, object()))
        with self.assertRaisesRegex(ValueError, "requires.*compact-2026-01-12"):
            non_beta_adapter.validate_requests(
                [
                    {
                        "custom_id": "context-test",
                        "params": {
                            **request(),
                            "context_management": {
                                "edits": [{"type": "compact_20260112"}]
                            },
                        },
                    }
                ]
            )
        with self.assertRaisesRegex(ValueError, "on-demand"):
            adapter.validate_requests(
                [
                    {
                        "custom_id": "on-demand-test",
                        "params": {**request(), "compaction": {"type": "summarize"}},
                    }
                ]
            )

    def test_anthropic_compaction_result_blocks_preserve_order_and_multiplicity(
        self,
    ) -> None:
        raw_result = {
            "type": "succeeded",
            "message": {
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "compaction",
                        "content": "private summary one",
                        "signature": "opaque-signature-one",
                        "encrypted_content": "opaque-encrypted-one",
                        "tool_changes": [
                            {
                                "type": "tool_addition",
                                "tool": {"type": "tool_reference", "name": "lookup"},
                            }
                        ],
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_one",
                        "name": "lookup",
                        "input": {"query": "sample"},
                    },
                    {
                        "type": "compaction",
                        "content": "private summary two",
                        "signature": "opaque-signature-two",
                    },
                ],
                "stop_reason": "tool_use",
            },
        }

        normalized = _normalize_anthropic_result(raw_result)
        parsed = parse_batch_result(normalized)

        expected_blocks = [
            {"type": "text", "text": "before"},
            {
                "type": "compaction",
                "content": "private summary one",
                "signature": "opaque-signature-one",
                "encrypted_content": "opaque-encrypted-one",
                "tool_changes": [
                    {
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": "lookup"},
                    }
                ],
            },
            {
                "type": "tool_use",
                "id": "toolu_one",
                "name": "lookup",
                "input": {"query": "sample"},
            },
            {
                "type": "compaction",
                "content": "private summary two",
                "signature": "opaque-signature-two",
            },
        ]
        self.assertEqual(normalized["type"], "succeeded")
        self.assertEqual(parsed["type"], "succeeded")
        self.assertEqual(
            successful_result(parsed)["message"]["content"], expected_blocks
        )

    def test_anthropic_failed_compaction_block_round_trips_as_noop(self) -> None:
        failed_block = {
            "type": "compaction",
            "content": None,
            "signature": None,
            "encrypted_content": "opaque-encrypted",
            "tool_changes": [],
        }
        self.assertEqual(
            _normalize_anthropic_result(
                {
                    "type": "succeeded",
                    "message": {
                        "content": [failed_block],
                        "stop_reason": "end_turn",
                    },
                }
            ),
            {
                "type": "succeeded",
                "message": {
                    "content": [failed_block],
                    "stop_reason": "end_turn",
                },
            },
        )

    def test_anthropic_compaction_blocks_reject_invalid_optional_field_types(
        self,
    ) -> None:
        for block in (
            {"type": "compaction", "content": 42, "signature": "opaque"},
            {"type": "compaction", "content": "summary", "signature": 42},
            {"type": "compaction", "encrypted_content": 42},
            {"type": "compaction", "tool_changes": "not-a-list"},
        ):
            with self.subTest(block_type=block), self.assertRaises(BatchProtocolError):
                _normalize_anthropic_result(
                    {
                        "type": "succeeded",
                        "message": {"content": [block], "stop_reason": "end_turn"},
                    }
                )

    def test_anthropic_on_demand_compaction_stop_reason_is_unsupported(self) -> None:
        self.assertEqual(
            _normalize_anthropic_result(
                {
                    "type": "succeeded",
                    "message": {"content": [], "stop_reason": "compaction"},
                }
            ),
            {
                "type": "errored",
                "error": {
                    "code": "unsupported_stop_reason",
                    "message": "Anthropic batch result has an unsupported stop reason",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
