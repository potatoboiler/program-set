"""Durable provider-profile storage tests; no SDK clients or APIs are used."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from program_set.batch_transport import (
    BatchProtocolError,
    BatchTransport,
    ProviderProfileNotFoundError,
)
from program_set.provider_profile import ProviderProfile


def test_provider_profile_upsert_survives_database_reopen(tmp_path: Path) -> None:
    database = tmp_path / "profiles.sqlite"
    transport = BatchTransport(None, database)
    transport.save_provider_profile(
        ProviderProfile(
            profile_id="production-anthropic",
            provider="anthropic",
            config={"betas": ["compact-2026-01-12"]},
        )
    )
    transport.save_provider_profile(
        ProviderProfile(
            profile_id="production-anthropic",
            provider="anthropic",
            config={"betas": ["compact-2026-01-12", "another-beta"]},
        )
    )
    transport.close()

    reopened = BatchTransport(None, database)
    try:
        profile = reopened.get_provider_profile("production-anthropic")
        assert profile == ProviderProfile(
            profile_id="production-anthropic",
            provider="anthropic",
            config={"betas": ["compact-2026-01-12", "another-beta"]},
        )
        assert reopened.list_provider_profiles() == (profile,)
        assert reopened.flush() is None
    finally:
        reopened.close()


def test_profile_config_rejects_credentials_and_does_not_store_them(
    tmp_path: Path,
) -> None:
    for config in (
        {"api_key": "never-persist-this"},
        {"nested": {"access-token": "never-persist-this"}},
        {"credentials": {"password": "never-persist-this"}},
    ):
        with pytest.raises(ValueError, match="credential fields"):
            ProviderProfile("safe-profile", "anthropic", config)

    database = tmp_path / "safe-profile.sqlite"
    transport = BatchTransport(None, database)
    try:
        transport.save_provider_profile(
            ProviderProfile(
                "safe-profile", "anthropic", {"betas": ["compact-2026-01-12"]}
            )
        )
    finally:
        transport.close()

    with sqlite3.connect(database) as connection:
        (config_json,) = connection.execute(
            "SELECT config_json FROM provider_profiles WHERE profile_id = ?",
            ("safe-profile",),
        ).fetchone()
    assert config_json == '{"betas":["compact-2026-01-12"]}'
    assert "never-persist-this" not in config_json


def test_missing_and_malformed_provider_profiles_fail_clearly(
    tmp_path: Path,
) -> None:
    database = tmp_path / "invalid-profile.sqlite"
    transport = BatchTransport(None, database)
    try:
        with pytest.raises(ProviderProfileNotFoundError):
            transport.get_provider_profile("missing")
        transport.save_provider_profile(ProviderProfile("stored", "openai"))
    finally:
        transport.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE provider_profiles SET provider = 'unknown' WHERE profile_id = ?",
            ("stored",),
        )

    reopened = BatchTransport(None, database)
    try:
        with pytest.raises(BatchProtocolError, match="malformed"):
            reopened.get_provider_profile("stored")
    finally:
        reopened.close()
