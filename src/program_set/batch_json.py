"""Small JSON and SDK-object helpers shared by batch adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping


def field(value: object, name: str) -> object:
    """Read one field from a provider mapping or SDK-shaped object."""

    if isinstance(value, Mapping):
        for key, member in value.items():
            if key == name:
                return member
        return None
    return getattr(value, name, None)


def jsonable(value: object) -> object:
    """Convert an SDK response model to its JSON-compatible representation."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for identity and SQLite persistence."""

    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
