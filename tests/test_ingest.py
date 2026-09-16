"""Testes da ingestão: carregamento, split 1000/150 e gravação idempotente."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

import ingest
from config import CHUNK_OVERLAP, CHUNK_SIZE, Settings


def make_pages(text: str = "palavra " * 500) -> list[Document]:
    return [Document(page_content=text, metadata={"source": "document.pdf", "page": 0})]


class FakeStore:
    """Vector store falso que registra o que foi gravado."""

    def __init__(self) -> None:
        """Inicializa o registro do que foi gravado."""
        self.documents: list[Document] = []
        self.ids: list[str] = []

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """Registra documentos e ids recebidos."""
        self.documents = documents
        self.ids = ids


def test_should_split_documents_in_chunks_of_1000_with_overlap_150() -> None:
    chunks = ingest.split_documents(make_pages())

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 150)


def test_should_keep_source_metadata_and_drop_empty_values() -> None:
    pages = [
        Document(
            page_content="conteúdo curto",
            metadata={"source": "document.pdf", "page": 2, "producer": ""},
        )
    ]

    chunks = ingest.split_documents(pages)

    assert chunks[0].metadata == {"source": "document.pdf", "page": 2, "start_index": 0}


def test_should_generate_deterministic_ids_across_independent_splits() -> None:
    primeira = ingest.build_chunk_ids(ingest.split_documents(make_pages()))
    segunda = ingest.build_chunk_ids(ingest.split_documents(make_pages()))

    assert primeira == segunda
    assert all(len(chunk_id) == 64 for chunk_id in primeira)


def test_should_derive_ids_from_sha256_of_source_page_offset_and_content() -> None:
    import hashlib

    chunk = Document(
        page_content="conteúdo curto",
        metadata={"source": "document.pdf", "page": 2, "start_index": 17},
    )
    esperado = hashlib.sha256("document.pdf|2|17|conteúdo curto".encode()).hexdigest()

    assert ingest.build_chunk_ids([chunk]) == [esperado]


def test_should_generate_different_ids_for_same_text_from_different_sources() -> None:
    def chunk_from(source: str) -> Document:
        return Document(
            page_content="mesmo texto",
            metadata={"source": source, "page": 0, "start_index": 0},
        )

    [id_a] = ingest.build_chunk_ids([chunk_from("a.pdf")])
    [id_b] = ingest.build_chunk_ids([chunk_from("b.pdf")])

    assert id_a != id_b


def test_should_keep_ids_stable_for_later_pages_when_an_earlier_page_changes() -> None:
    def pages(primeira: str) -> list[Document]:
        return [
            Document(page_content=primeira, metadata={"source": "d.pdf", "page": 0}),
            Document(
                page_content="conteúdo estável",
                metadata={"source": "d.pdf", "page": 1},
            ),
        ]

    antes = ingest.build_chunk_ids(ingest.split_documents(pages("texto original")))
    depois = ingest.build_chunk_ids(
        ingest.split_documents(pages("texto original bem mais longo depois da edição"))
    )

    assert antes[-1] == depois[-1]


def test_should_generate_distinct_ids_for_different_chunks() -> None:
    ids = ingest.build_chunk_ids(ingest.split_documents(make_pages()))

    assert len(set(ids)) == len(ids)


def test_should_raise_when_pdf_file_does_not_exist() -> None:
    with pytest.raises(FileNotFoundError, match="PDF não encontrado"):
        ingest.load_pdf(Path("/caminho/que/nao/existe.pdf"))


def test_should_load_every_page_of_a_real_pdf() -> None:
    pages = ingest.load_pdf(Path("document.pdf"))

    assert pages
    assert all(page.metadata["source"].endswith("document.pdf") for page in pages)


def test_should_store_every_chunk_with_its_id(
    monkeypatch: pytest.MonkeyPatch, openai_settings: Settings
) -> None:
    store = FakeStore()
    monkeypatch.setattr(ingest, "get_settings", lambda: openai_settings)
    monkeypatch.setattr(ingest, "load_pdf", lambda _path: make_pages())
    monkeypatch.setattr(ingest, "get_vector_store", lambda _settings: store)

    total = ingest.ingest_pdf()

    assert total == len(store.documents)
    assert len(store.ids) == total


def test_should_raise_when_pdf_has_no_extractable_text(
    monkeypatch: pytest.MonkeyPatch, openai_settings: Settings
) -> None:
    monkeypatch.setattr(ingest, "get_settings", lambda: openai_settings)
    monkeypatch.setattr(ingest, "load_pdf", lambda _path: [])

    with pytest.raises(RuntimeError, match="Nenhum texto extraído"):
        ingest.ingest_pdf()


def test_should_overlap_150_characters_between_consecutive_chunks() -> None:
    texto = " ".join(f"palavra{indice:04d}" for indice in range(1200))
    chunks = ingest.split_documents(
        [Document(page_content=texto, metadata={"source": "d.pdf", "page": 0})]
    )

    assert len(chunks) > 2
    primeiro, segundo = chunks[0].page_content, chunks[1].page_content
    sobreposicao = next(
        (
            primeiro[-tamanho:]
            for tamanho in range(CHUNK_OVERLAP, 0, -1)
            if segundo.startswith(primeiro[-tamanho:])
        ),
        "",
    )

    assert len(sobreposicao) > CHUNK_OVERLAP // 2
