"""Small caller-side helpers for resuming completed batch-backed Strands work."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from strands.interrupt import Interrupt
from strands.types.interrupt import InterruptResponseContent

from program_set.batch_transport import BatchTransport, SubmissionUncertainError

_BATCH_INTERRUPT_NAME = "batch-response-pending"


def ready_batch_interrupt_responses(
    interrupts: Sequence[Interrupt], transport: BatchTransport
) -> list[InterruptResponseContent]:
    """Build responses only for this package's durably succeeded batch requests.

    Only interrupts carrying this model's source marker are considered. Foreign
    interrupts and queued/submitting/submitted requests are left without a
    response. A ``submit_unknown`` request raises
    :class:`SubmissionUncertainError` with its durable submission and request IDs
    so the caller can investigate safely.
    Unknown IDs, mismatched custom IDs, and terminal failures are surfaced to the
    caller; this helper does not flush, poll, invoke, or decide whether to retry.
    """

    responses: list[InterruptResponseContent] = []
    for interrupt in interrupts:
        if interrupt.name != _BATCH_INTERRUPT_NAME:
            continue
        if (
            not isinstance(interrupt.reason, dict)
            or interrupt.reason.get("source") != "program-set.deferred-batch-model"
        ):
            continue
        request_id = interrupt.reason.get("request_id")
        custom_id = interrupt.reason.get("custom_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("batch interrupt has no valid request_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"batch interrupt {request_id!r} has no valid custom_id")

        record = transport.get_request(request_id)
        if record.custom_id != custom_id:
            raise ValueError(
                f"batch interrupt {request_id!r} custom_id {custom_id!r} "
                f"does not match durable custom_id {record.custom_id!r}"
            )
        if record.status == "submit_unknown":
            if record.submission_id is None:
                raise RuntimeError(
                    f"batch request {request_id!r} has unknown submission status "
                    "but no durable submission_id; inspect transport state"
                )
            raise SubmissionUncertainError(record.submission_id, (request_id,))
        if record.status in {"queued", "submitting", "submitted"}:
            continue
        if (
            record.status != "succeeded"
            or record.result_status != "succeeded"
            or record.result is None
            or record.result["type"] != "succeeded"
        ):
            raise RuntimeError(
                f"batch request {request_id!r} (custom_id {custom_id!r}) "
                f"completed as {record.status!r} with result "
                f"status {record.result_status!r}"
            )

        response = cast(
            InterruptResponseContent,
            {
                "interruptResponse": {
                    "interruptId": interrupt.id,
                    "response": {"request_id": request_id, "ready": True},
                }
            },
        )
        responses.append(response)
    return responses
