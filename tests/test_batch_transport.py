from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import anthropic
import httpx2

from program_set.batch_transport import (
    AnthropicBatchClient,
    BatchInfo,
    BatchProtocolError,
    BatchTransport,
    MessagesBatchRequest,
    RequestConflictError,
    SubmissionUncertainError,
)


def request(text: str = "hello") -> dict[str, object]:
    return {
        "model": "claude-test",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": text}],
    }


class FakeBatchClient:
    def __init__(self) -> None:
        self.submissions: list[list[MessagesBatchRequest]] = []
        self.batch = BatchInfo("msgbatch_test", "in_progress")
        self.results: list[object] = []
        self.fail_create = False

    def create_batch(self, requests: Sequence[MessagesBatchRequest]) -> BatchInfo:
        self.submissions.append(list(requests))
        if self.fail_create:
            raise TimeoutError("simulated lost response")
        return self.batch

    def retrieve_batch(self, batch_id: str) -> BatchInfo:
        assert batch_id == self.batch.batch_id
        return self.batch

    def iter_batch_results(self, batch_id: str) -> list[object]:
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
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                },
            },
        ]
        self.assertEqual(
            self.transport.poll("msgbatch_test").processing_status, "ended"
        )

        first_record = self.transport.get_request(first.request_id)
        second_record = self.transport.get_request(second.request_id)
        self.assertEqual(first_record.status, "succeeded")
        self.assertEqual(first_record.result_status, "succeeded")
        self.assertEqual(first_record.result["message"]["id"], "msg_1")
        self.assertEqual(first_record.metadata, {"job": "alpha"})
        self.assertEqual(second_record.status, "errored")
        self.assertEqual(second_record.result_status, "errored")

    def test_batch_registry_and_results_survive_reopening_database(self) -> None:
        self.transport.enqueue(request(), request_id="durable", metadata={"item": "42"})
        self.transport.flush()
        self.client.batch = BatchInfo("msgbatch_test", "ended")
        self.client.results = [
            {"custom_id": "durable", "result": {"type": "canceled"}}
        ]
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

    def test_anthropic_custom_id_and_unsupported_params_are_rejected(self) -> None:
        for index, custom_id in enumerate(
            ("", "contains.dot", "has space", "x" * 65, "é")
        ):
            with self.subTest(custom_id=custom_id), self.assertRaises(ValueError):
                self.transport.enqueue(
                    request(), request_id=f"invalid-{index}", custom_id=custom_id
                )

        accepted = self.transport.enqueue(
            request(), request_id="valid", custom_id="a" * 64
        )
        self.assertEqual(accepted.custom_id, "a" * 64)

        for unsupported in ({"stream": True}, {"speed": "fast"}):
            params = {**request(), **unsupported}
            with self.subTest(params=unsupported), self.assertRaises(ValueError):
                self.transport.enqueue(
                    params,
                    request_id=f"unsupported-{len(unsupported)}-{next(iter(unsupported))}",
                )

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
            adapter = AnthropicBatchClient(sdk_client)
            with self.assertRaises(anthropic.APIStatusError):
                adapter.create_batch(
                    [
                        {
                            "custom_id": "sdk-test",
                            "params": request(),
                        }
                    ]
                )
            self.assertEqual(attempts, 1)
        finally:
            sdk_client.close()

    def test_result_with_unknown_custom_id_is_rejected(self) -> None:
        self.transport.enqueue(request(), request_id="known")
        self.transport.flush()
        self.client.batch = BatchInfo("msgbatch_test", "ended")
        self.client.results = [
            {"custom_id": "other", "result": {"type": "canceled"}}
        ]
        with self.assertRaises(BatchProtocolError):
            self.transport.poll("msgbatch_test")

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
                self.sent: list[MessagesBatchRequest] = []

            def create(self, *, requests: list[MessagesBatchRequest]) -> object:
                self.sent = requests
                return {"id": "remote-id", "processing_status": "in_progress"}

            def retrieve(self, batch_id: str) -> object:
                return {"id": batch_id, "processing_status": "ended"}

            def results(self, batch_id: str) -> list[object]:
                return [{"custom_id": "id", "result": {"type": "canceled"}}]

        class Messages:
            def __init__(self) -> None:
                self.batches = Batches()

        class SDKClient:
            def __init__(self) -> None:
                self.messages = Messages()

        sdk_client = SDKClient()
        adapter = AnthropicBatchClient(sdk_client)
        payload: MessagesBatchRequest = {"custom_id": "id", "params": request()}
        self.assertEqual(adapter.create_batch([payload]).batch_id, "remote-id")
        self.assertEqual(sdk_client.messages.batches.sent, [payload])
        self.assertEqual(adapter.retrieve_batch("remote-id").processing_status, "ended")
        self.assertEqual(len(list(adapter.iter_batch_results("remote-id"))), 1)


if __name__ == "__main__":
    unittest.main()
