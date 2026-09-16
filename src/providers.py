"""Fábrica de embeddings, LLM e vector store conforme o provider configurado."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_postgres import PGVector

from config import (
    LLM_TEMPERATURE,
    REQUEST_TIMEOUT_SECONDS,
    Settings,
    get_settings,
)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Cria o modelo de embeddings do provider configurado.

    Args:
        settings: Configuração já carregada; usa a global quando omitida.

    Returns:
        Implementação de `Embeddings` pronta para uso.
    """
    settings = settings or get_settings()

    if settings.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.api_key,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.api_key,
    )


def _openai_reasoning_kwargs(model: str) -> dict[str, str]:
    """Desliga o raciocínio nos modelos que, sem isso, ignorariam `temperature`.

    A família `gpt-5` só aceita `temperature` diferente de 1 quando o esforço de
    raciocínio é explicitamente `none`; caso contrário `langchain_openai` remove o
    parâmetro no validator e a resposta deixa de ser determinística.

    Args:
        model: Nome do modelo de linguagem configurado.

    Returns:
        `{"reasoning_effort": "none"}` quando necessário; dicionário vazio caso contrário.
    """
    name = model.lower()
    if name.startswith("gpt-5") and "chat" not in name:
        return {"reasoning_effort": "none"}
    return {}


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Cria o modelo de linguagem do provider configurado.

    Args:
        settings: Configuração já carregada; usa a global quando omitida.

    Returns:
        Chat model determinístico (temperatura zero) para responder as perguntas.
    """
    settings = settings or get_settings()

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=LLM_TEMPERATURE,
            timeout=REQUEST_TIMEOUT_SECONDS,
            openai_api_key=settings.api_key,
            **_openai_reasoning_kwargs(settings.llm_model),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=LLM_TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        google_api_key=settings.api_key,
    )


def get_vector_store(settings: Settings | None = None) -> PGVector:
    """Abre a collection pgVector usada na ingestão e na busca.

    Args:
        settings: Configuração já carregada; usa a global quando omitida.

    Returns:
        Vector store conectado ao Postgres definido em `DATABASE_URL`.
    """
    settings = settings or get_settings()

    return PGVector(
        embeddings=get_embeddings(settings),
        collection_name=settings.pg_vector_collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )
