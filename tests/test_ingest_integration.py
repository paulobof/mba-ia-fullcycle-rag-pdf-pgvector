"""Teste de integração da ingestão contra um Postgres real com pgVector.

Fora do gate padrão (`-m 'not integration'`). Para rodar:
    make db-up && uv run pytest -m integration
"""

import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import ingest
import providers
from conftest import SettingsFactory

pytestmark = pytest.mark.integration

DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
)
EMBEDDING_DIMENSIONS = 128


def count_vectors(store: object) -> int:
    """Conta os vetores presentes na collection de teste.

    Args:
        store: Vector store já conectado.

    Returns:
        Quantidade de embeddings gravados.
    """
    from sqlalchemy import text

    with store._make_sync_session() as session:  # type: ignore[attr-defined]
        collection = store.get_collection(session)  # type: ignore[attr-defined]
        result = session.execute(
            text(
                "SELECT count(*) FROM langchain_pg_embedding "
                "WHERE collection_id = :collection_id"
            ),
            {"collection_id": collection.uuid},
        )
        return int(result.scalar_one())


def test_should_not_duplicate_vectors_when_ingesting_the_same_pdf_twice(
    monkeypatch: pytest.MonkeyPatch, make_settings: SettingsFactory
) -> None:
    pytest.importorskip("psycopg")
    settings = make_settings(
        openai_api_key="sk-test",
        database_url=DATABASE_URL,
        pg_vector_collection_name="teste_idempotencia",
    )
    monkeypatch.setattr(
        providers,
        "get_embeddings",
        lambda _settings=None: DeterministicFakeEmbedding(size=EMBEDDING_DIMENSIONS),
    )

    pages = [
        Document(page_content="texto " * 400, metadata={"source": "d.pdf", "page": 0})
    ]
    chunks = ingest.split_documents(pages)
    ids = ingest.build_chunk_ids(chunks)

    store = providers.get_vector_store(settings)
    try:
        store.delete_collection()
        store.create_collection()

        store.add_documents(documents=chunks, ids=ids)
        apos_primeira = count_vectors(store)
        store.add_documents(documents=chunks, ids=ids)
        apos_segunda = count_vectors(store)

        store.delete_collection()
    finally:
        store._engine.dispose()  # type: ignore[union-attr]

    assert apos_primeira == len(chunks)
    assert apos_segunda == apos_primeira, (
        "reingestão duplicou vetores em vez de fazer upsert"
    )
