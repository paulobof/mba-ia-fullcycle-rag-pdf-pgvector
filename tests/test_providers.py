"""Testes da fábrica de provider (embeddings, LLM e vector store)."""

from typing import Any

import pytest

import providers
from config import LLM_TEMPERATURE, REQUEST_TIMEOUT_SECONDS, Settings
from conftest import SettingsFactory


def test_should_build_openai_embeddings_with_configured_model(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(
        openai_api_key="sk-test", openai_embedding_model="text-embedding-3-large"
    )

    embeddings = providers.get_embeddings(settings)

    assert type(embeddings).__name__ == "OpenAIEmbeddings"
    assert embeddings.model == "text-embedding-3-large"  # type: ignore[attr-defined]


def test_should_build_openai_llm_with_the_configured_model(
    openai_settings: Settings,
) -> None:
    llm = providers.get_llm(openai_settings)

    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == openai_settings.llm_model  # type: ignore[attr-defined]


def test_should_pin_temperature_to_zero_for_deterministic_answers(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(openai_api_key="sk-test", openai_llm_model="gpt-4o-mini")

    llm = providers.get_llm(settings)

    assert LLM_TEMPERATURE == 0.0
    assert llm.temperature == 0.0  # type: ignore[attr-defined]


def test_should_keep_zero_temperature_on_gpt5_models_by_disabling_reasoning(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(openai_api_key="sk-test", openai_llm_model="gpt-5.6-luna")

    llm = providers.get_llm(settings)

    assert llm.temperature == 0.0  # type: ignore[attr-defined]
    assert llm.reasoning_effort == "none"  # type: ignore[attr-defined]


def test_should_not_disable_reasoning_on_models_that_accept_temperature(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(openai_api_key="sk-test", openai_llm_model="gpt-4o-mini")

    llm = providers.get_llm(settings)

    assert llm.reasoning_effort is None  # type: ignore[attr-defined]


def test_should_bound_llm_requests_with_a_timeout(
    make_settings: SettingsFactory, gemini_settings: Settings
) -> None:
    openai_llm = providers.get_llm(
        make_settings(openai_api_key="sk-test", openai_llm_model="gpt-4o-mini")
    )
    gemini_llm = providers.get_llm(gemini_settings)

    assert openai_llm.request_timeout == REQUEST_TIMEOUT_SECONDS  # type: ignore[attr-defined]
    assert gemini_llm.timeout == REQUEST_TIMEOUT_SECONDS  # type: ignore[attr-defined]


def test_should_build_gemini_embeddings_when_provider_is_gemini(
    gemini_settings: Settings,
) -> None:
    embeddings = providers.get_embeddings(gemini_settings)

    assert type(embeddings).__name__ == "GoogleGenerativeAIEmbeddings"
    assert embeddings.model.endswith(gemini_settings.embedding_model)  # type: ignore[attr-defined]


def test_should_build_gemini_llm_when_provider_is_gemini(
    gemini_settings: Settings,
) -> None:
    llm = providers.get_llm(gemini_settings)

    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith(gemini_settings.llm_model)  # type: ignore[attr-defined]


def test_should_open_vector_store_with_configured_collection(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    captured: dict[str, Any] = {}

    class FakePGVector:
        """Substituto do PGVector que captura os argumentos de construção."""

        def __init__(self, **kwargs: Any) -> None:
            """Captura os argumentos recebidos."""
            captured.update(kwargs)

    monkeypatch.setattr(providers, "PGVector", FakePGVector)
    monkeypatch.setattr(providers, "get_embeddings", lambda _settings: "fake-embeddings")
    settings = make_settings(
        openai_api_key="sk-test",
        pg_vector_collection_name="minha_collection",
        database_url="postgresql+psycopg://user:pass@localhost:5432/rag",
    )

    providers.get_vector_store(settings)

    assert captured["collection_name"] == "minha_collection"
    assert captured["connection"] == "postgresql+psycopg://user:pass@localhost:5432/rag"
    assert captured["use_jsonb"] is True
    assert captured["embeddings"] == "fake-embeddings"


def test_should_fall_back_to_global_settings_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch, openai_settings: Settings
) -> None:
    monkeypatch.setattr(providers, "get_settings", lambda: openai_settings)

    llm = providers.get_llm()

    assert type(llm).__name__ == "ChatOpenAI"
