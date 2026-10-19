"""Durable, non-secret configuration for constructing batch adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

ProviderName = Literal["anthropic", "openai"]
_PROVIDERS = {"anthropic", "openai"}
_SECRET_CONFIG_KEY_PARTS = (
    "apikey",
    "secret",
    "token",
    "credential",
    "password",
    "authorization",
    "auth",
)


@dataclass(frozen=True)
class ProviderProfile:
    """Adapter configuration safe to store alongside batch state.

    Credentials are intentionally not represented here. Pass SDK clients to
    ``provider_factory.build_batch_client`` from application-owned secret
    handling instead.
    """

    profile_id: str
    provider: ProviderName
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.provider, str) or self.provider not in _PROVIDERS:
            raise ValueError(f"unsupported provider {self.provider!r}")
        if not isinstance(self.config, Mapping):
            raise TypeError("provider profile config must be a mapping")
        copied_config = dict(self.config)
        _reject_credential_fields(copied_config)
        if self.provider == "anthropic":
            unknown = set(copied_config) - {"betas"}
            if unknown:
                raise ValueError(
                    "unsupported Anthropic provider profile config keys: "
                    f"{sorted(unknown)!r}"
                )
            betas = copied_config.get("betas")
            if betas is not None and (
                not isinstance(betas, Sequence)
                or isinstance(betas, (str, bytes))
                or any(not isinstance(beta, str) or not beta for beta in betas)
            ):
                raise ValueError(
                    "Anthropic provider profile betas must be a sequence of "
                    "non-empty strings"
                )
        elif copied_config:
            raise ValueError("OpenAI provider profiles do not accept config keys")
        object.__setattr__(self, "config", MappingProxyType(copied_config))


def _reject_credential_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("provider profile config keys must be strings")
            normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
            if any(part in normalized_key for part in _SECRET_CONFIG_KEY_PARTS):
                raise ValueError(
                    "provider profile config must not contain credential fields"
                )
            _reject_credential_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credential_fields(item)
