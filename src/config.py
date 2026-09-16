"""Configuração tipada da aplicação, validada com Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self, assert_never

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "gemini"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 10
MIN_RELEVANCE = 0.35
LLM_TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 30


class Settings(BaseSettings):
    """Variáveis de ambiente necessárias para ingestão e busca.

    Attributes:
        provider: Provider usado para embeddings e LLM.
        database_url: URL SQLAlchemy do Postgres com a extensão pgVector.
        pdf_path: Caminho do PDF a ser ingerido.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    provider: Provider = "openai"

    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-5.6-luna"

    google_api_key: SecretStr | None = None
    google_embedding_model: str = "gemini-embedding-001"
    google_llm_model: str = "gemini-2.5-flash-lite"

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/rag",
        min_length=1,
    )
    pg_vector_collection_name: str = Field(default="desafio_rag", min_length=1)
    pdf_path: Path = Path("document.pdf")

    @model_validator(mode="after")
    def check_provider_credentials(self) -> Self:
        """Garante que a chave do provider selecionado está preenchida.

        Returns:
            A própria instância validada.

        Raises:
            ValueError: Se a chave do provider escolhido estiver ausente.
        """
        required = {
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "gemini": ("GOOGLE_API_KEY", self.google_api_key),
        }
        env_name, secret = required[self.provider]
        if secret is None or not secret.get_secret_value().strip():
            message = (
                f"{env_name} é obrigatória quando PROVIDER='{self.provider}'. "
                "Copie .env.example para .env e preencha os valores."
            )
            raise ValueError(message)
        return self

    @property
    def embedding_model(self) -> str:
        """Modelo de embeddings do provider ativo."""
        match self.provider:
            case "openai":
                return self.openai_embedding_model
            case "gemini":
                return self.google_embedding_model
            case _ as unreachable:  # pragma: no cover - exaustividade garantida por mypy
                assert_never(unreachable)

    @property
    def llm_model(self) -> str:
        """Modelo de linguagem do provider ativo."""
        match self.provider:
            case "openai":
                return self.openai_llm_model
            case "gemini":
                return self.google_llm_model
            case _ as unreachable:  # pragma: no cover - exaustividade garantida por mypy
                assert_never(unreachable)

    @property
    def api_key(self) -> SecretStr:
        """Chave de API do provider ativo, já validada."""
        key = self.openai_api_key if self.provider == "openai" else self.google_api_key
        if key is None:  # pragma: no cover - garantido por check_provider_credentials
            message = f"Chave de API ausente para o provider '{self.provider}'."
            raise RuntimeError(message)
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Carrega (e memoriza) as configurações da aplicação.

    Returns:
        Instância única de `Settings`.
    """
    return Settings()
