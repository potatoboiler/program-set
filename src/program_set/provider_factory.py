"""Construct provider batch adapters from durable, non-secret profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from program_set.anthropic_batch import AnthropicBatchClient, _AnthropicSDKClient
from program_set.batch_transport import BatchClient
from program_set.openai_batch_transport import OpenAIBatchClient, _OpenAIClient
from program_set.provider_profile import ProviderProfile


def build_batch_client(
    profile: ProviderProfile,
    *,
    anthropic_client: object | None = None,
    openai_client: object | None = None,
) -> BatchClient:
    """Build an adapter from profile config and an application-owned SDK client.

    This factory never constructs SDK clients or handles credentials. Anthropic
    supports the non-secret ``betas`` option; OpenAI currently has no profile
    options. Unknown provider/config combinations fail before adapter creation.
    """

    if profile.provider == "anthropic":
        _reject_unknown_config(profile.config, {"betas"}, "Anthropic")
        betas = profile.config.get("betas")
        if betas is not None and (
            not isinstance(betas, Sequence)
            or isinstance(betas, (str, bytes))
            or any(not isinstance(beta, str) or not beta for beta in betas)
        ):
            raise ValueError(
                "Anthropic provider profile betas must be a sequence of "
                "non-empty strings"
            )
        if anthropic_client is None:
            raise ValueError(
                "an injected Anthropic SDK client is required for this profile"
            )
        return AnthropicBatchClient(
            cast(_AnthropicSDKClient, anthropic_client), betas=betas
        )

    if profile.provider == "openai":
        _reject_unknown_config(profile.config, set(), "OpenAI")
        if openai_client is None:
            raise ValueError(
                "an injected OpenAI SDK client is required for this profile"
            )
        return OpenAIBatchClient(cast(_OpenAIClient, openai_client))

    raise ValueError(f"unsupported provider {profile.provider!r}")


def _reject_unknown_config(
    config: Mapping[str, object], supported: set[str], provider: str
) -> None:
    unknown = set(config) - supported
    if unknown:
        raise ValueError(
            f"unsupported {provider} provider profile config keys: {sorted(unknown)!r}"
        )
