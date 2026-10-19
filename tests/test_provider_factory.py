"""Offline provider-factory and profile-restart tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from program_set.anthropic_batch import AnthropicBatchClient
from program_set.batch_transport import BatchTransport
from program_set.openai_batch_transport import OpenAIBatchClient
from program_set.provider_factory import build_batch_client
from program_set.provider_profile import ProviderProfile


class FakeBatches:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def create(
        self,
        *,
        requests: Sequence[Any],
        betas: list[str] | None = None,
    ) -> object:
        assert requests
        self.calls.append(("create", betas))
        return {"id": "fake-batch", "processing_status": "in_progress"}

    def retrieve(self, batch_id: str, *, betas: list[str] | None = None) -> object:
        self.calls.append((f"retrieve:{batch_id}", betas))
        return {"id": batch_id, "processing_status": "ended"}

    def results(self, batch_id: str, *, betas: list[str] | None = None) -> list[object]:
        self.calls.append((f"results:{batch_id}", betas))
        return []


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = SimpleNamespace(batches=FakeBatches())
        self.beta = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches()))

    def with_options(self, *, max_retries: int) -> FakeAnthropicClient:
        assert max_retries == 0
        return self


def _request() -> dict[str, object]:
    return {
        "model": "claude-test",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_factory_constructs_non_beta_anthropic_and_openai_adapters() -> None:
    anthropic_client = FakeAnthropicClient()
    anthropic_adapter = build_batch_client(
        ProviderProfile("plain-anthropic", "anthropic"),
        anthropic_client=anthropic_client,
    )
    assert isinstance(anthropic_adapter, AnthropicBatchClient)
    assert anthropic_adapter._betas is None
    anthropic_adapter.create_batch([{"custom_id": "plain", "params": _request()}])
    assert anthropic_client.messages.batches.calls == [("create", None)]

    openai_adapter = build_batch_client(
        ProviderProfile("plain-openai", "openai"), openai_client=object()
    )
    assert isinstance(openai_adapter, OpenAIBatchClient)


def test_saved_profile_reconstructs_beta_adapter_after_transport_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "provider-restart.sqlite"
    profile = ProviderProfile(
        "compaction-anthropic",
        "anthropic",
        {"betas": ["compact-2026-01-12"]},
    )
    initial_transport = BatchTransport(None, database)
    initial_transport.save_provider_profile(profile)
    initial_transport.close()

    restarted_transport = BatchTransport(None, database)
    sdk_client = FakeAnthropicClient()
    try:
        restored_profile = restarted_transport.get_provider_profile(
            "compaction-anthropic"
        )
        adapter = build_batch_client(restored_profile, anthropic_client=sdk_client)
        assert isinstance(adapter, AnthropicBatchClient)
        assert adapter._betas == ("compact-2026-01-12",)
        restarted_transport.set_client(adapter)
        restarted_transport.enqueue(_request(), request_id="restored")
        assert restarted_transport.flush() is not None
        assert sdk_client.beta.messages.batches.calls == [
            ("create", ["compact-2026-01-12"])
        ]
        assert sdk_client.messages.batches.calls == []
    finally:
        restarted_transport.close()


def test_factory_rejects_unknown_provider_missing_clients_and_bad_config() -> None:
    unknown_profile = cast(
        ProviderProfile,
        SimpleNamespace(provider="unknown", config={}),
    )
    with pytest.raises(ValueError, match="unsupported provider"):
        build_batch_client(unknown_profile)

    with pytest.raises(ValueError, match="Anthropic SDK client is required"):
        build_batch_client(ProviderProfile("anthropic", "anthropic"))
    with pytest.raises(ValueError, match="OpenAI SDK client is required"):
        build_batch_client(ProviderProfile("openai", "openai"))

    invalid_anthropic_config = cast(
        ProviderProfile,
        SimpleNamespace(provider="anthropic", config={"region": "test"}),
    )
    with pytest.raises(ValueError, match="unsupported Anthropic.*config keys"):
        build_batch_client(
            invalid_anthropic_config,
            anthropic_client=FakeAnthropicClient(),
        )
    for value in ("not-a-list", ["", "good"], ["good", 5]):
        with pytest.raises(ValueError, match="sequence of non-empty strings"):
            build_batch_client(
                cast(
                    ProviderProfile,
                    SimpleNamespace(provider="anthropic", config={"betas": value}),
                ),
                anthropic_client=FakeAnthropicClient(),
            )
    with pytest.raises(ValueError, match="unsupported OpenAI.*config keys"):
        build_batch_client(
            cast(
                ProviderProfile,
                SimpleNamespace(provider="openai", config={"betas": ["beta"]}),
            ),
            openai_client=object(),
        )
