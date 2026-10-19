"""Small type-safe constructors around intentionally dynamic test data."""

from __future__ import annotations

from typing import cast

from strands.types.content import Messages

from program_set.batch_result import (
    BatchResult,
    NormalizedMessage,
    SucceededBatchResult,
)


def strands_messages(value: object) -> Messages:
    """Treat explicitly supplied test data as a Strands message boundary."""

    return cast(Messages, value)


def succeeded(message: NormalizedMessage) -> SucceededBatchResult:
    return {"type": "succeeded", "message": message}


def successful_result(result: BatchResult | None) -> SucceededBatchResult:
    assert result is not None
    assert result["type"] == "succeeded"
    return result
