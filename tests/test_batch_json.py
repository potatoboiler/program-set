from __future__ import annotations

import pytest

from program_set.batch_json import canonical_json, field, jsonable


def test_field_reads_mapping_and_sdk_attributes() -> None:
    class SDKValue:
        value = "sdk"

    assert field({"value": "mapping"}, "value") == "mapping"
    assert field(SDKValue(), "value") == "sdk"
    assert field({}, "missing") is None


def test_jsonable_prefers_model_dump_then_to_dict() -> None:
    class ModelDumpValue:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"model": True}

        def to_dict(self) -> dict[str, object]:
            raise AssertionError("model_dump should take precedence")

    class DictValue:
        def to_dict(self) -> dict[str, object]:
            return {"dict": True}

    assert jsonable(ModelDumpValue()) == {"model": True}
    assert jsonable(DictValue()) == {"dict": True}
    assert jsonable({"plain": True}) == {"plain": True}


def test_canonical_json_is_sorted_compact_and_rejects_nan() -> None:
    assert canonical_json({"z": "ø", "a": 1}) == '{"a":1,"z":"ø"}'
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"not_json": float("nan")})
