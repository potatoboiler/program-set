"""Opaque bridge for Anthropic compaction blocks through Strands signatures."""

from __future__ import annotations

import base64
import json

from program_set.batch_result import (
    BatchResultError,
    CompactionContentBlock,
    parse_compaction_block,
)

COMPACTION_SIGNATURE_PREFIX = "anthropic-compaction:v1:"


def encode_compaction_signature(block: CompactionContentBlock) -> str:
    """Encode a complete provider block for Strands' opaque signature field."""

    payload = json.dumps(
        block,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return COMPACTION_SIGNATURE_PREFIX + encoded


def decode_compaction_signature(value: str) -> CompactionContentBlock | None:
    """Decode an adapter-produced block, without including opaque data in errors."""

    if not value.startswith(COMPACTION_SIGNATURE_PREFIX):
        return None
    encoded = value[len(COMPACTION_SIGNATURE_PREFIX) :]
    padded = encoded + "=" * ((4 - len(encoded) % 4) % 4)
    try:
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
        block = json.loads(payload.decode("utf-8"))
        return parse_compaction_block(block)
    except BatchResultError, ValueError, UnicodeDecodeError:
        raise ValueError("invalid opaque Anthropic compaction signature") from None
