"""Testes da configuração tipada (Pydantic Settings)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import config
from config import Settings
from conftest import SettingsFactory


def test_should_use_openai_as_default_provider(openai_settings: Settings) -> None:
    assert openai_settings.provider == "openai"
    assert openai_settings.embedding_model == "text-embedding-3-small"
    assert openai_settings.llm_model == "gpt-5.6-luna"


def test_should_expose_gemini_models_when_provider_is_gemini(
    gemini_settings: Settings,
) -> None:
    assert gemini_settings.embedding_model == "gemini-embedding-001"
    assert gemini_settings.llm_model == "gemini-2.5-flash-lite"


def test_should_reject_openai_provider_without_api_key(
    make_settings: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_settings()


def test_should_reject_gemini_provider_without_api_key(
    make_settings: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        make_settings(provider="gemini")


def test_should_reject_blank_api_key(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_settings(openai_api_key="   ")


def test_should_reject_unsupported_provider(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValidationError):
        make_settings(provider="anthropic", openai_api_key="sk-test")


def test_should_reject_empty_collection_name(make_settings: SettingsFactory) -> None:
    with pytest.raises(ValidationError):
        make_settings(openai_api_key="sk-test", pg_vector_collection_name="")


def test_should_read_values_from_environment(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("PDF_PATH", "/tmp/outro.pdf")

    settings = make_settings()

    assert settings.llm_model == "gemini-3.5-flash-lite"
    assert settings.pdf_path == Path("/tmp/outro.pdf")


def test_should_keep_api_key_masked_in_repr(openai_settings: Settings) -> None:
    assert "sk-test" not in repr(openai_settings)
    assert openai_settings.api_key.get_secret_value() == "sk-test"


def test_should_be_immutable(openai_settings: Settings) -> None:
    with pytest.raises(ValidationError):
        openai_settings.provider = "gemini"  # type: ignore[misc]


def test_should_cache_settings_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert config.get_settings() is config.get_settings()
