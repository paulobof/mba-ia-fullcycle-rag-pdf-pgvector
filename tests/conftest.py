"""Fixtures compartilhadas: isolam os testes de variáveis de ambiente e de rede."""

from collections.abc import Callable, Iterator

import pytest

import config
from config import Settings

ENV_VARS = (
    "PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_LLM_MODEL",
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "GOOGLE_LLM_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)


SettingsFactory = Callable[..., Settings]


def _build_settings(**overrides: object) -> Settings:
    """Cria `Settings` ignorando o arquivo .env do repositório.

    Args:
        **overrides: Campos a sobrescrever explicitamente.

    Returns:
        Instância isolada de configuração.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def make_settings() -> SettingsFactory:
    """Fábrica de `Settings` isolada do arquivo .env."""
    return _build_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isola os testes do ambiente real e do arquivo `.env` do desenvolvedor."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def openai_settings() -> Settings:
    """Configuração mínima válida para o provider OpenAI."""
    return _build_settings(openai_api_key="sk-test")


@pytest.fixture
def gemini_settings() -> Settings:
    """Configuração mínima válida para o provider Gemini."""
    return _build_settings(provider="gemini", google_api_key="fake-key")
