"""Ingestão do PDF: carrega, divide em chunks e grava os vetores no pgVector."""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, get_settings
from providers import get_vector_store


def load_pdf(pdf_path: Path) -> list[Document]:
    """Carrega o PDF em documentos, uma página por documento.

    Args:
        pdf_path: Caminho do arquivo PDF.

    Returns:
        Lista de páginas do PDF.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = pdf_path.expanduser()
    if not path.is_file():
        message = f"PDF não encontrado em '{path}'. Ajuste PDF_PATH no .env."
        raise FileNotFoundError(message)
    return PyPDFLoader(str(path)).load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Divide os documentos em chunks de 1000 caracteres com overlap de 150.

    Args:
        documents: Páginas carregadas do PDF.

    Returns:
        Chunks com `start_index` na metadata e sem chaves vazias.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata = {
            key: value for key, value in chunk.metadata.items() if value not in ("", None)
        }
    return chunks


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Gera ids determinísticos para permitir reingestão sem duplicar vetores.

    Args:
        chunks: Chunks já divididos.

    Returns:
        Ids hexadecimais derivados de origem, página, offset e conteúdo.
    """
    ids: list[str] = []
    for position, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        offset = chunk.metadata.get("start_index", position)
        seed = f"{source}|{page}|{offset}|{chunk.page_content}"
        ids.append(hashlib.sha256(seed.encode("utf-8")).hexdigest())
    return ids


def ingest_pdf() -> int:
    """Executa a ingestão completa do PDF no banco vetorial.

    Returns:
        Quantidade de chunks gravados.

    Raises:
        RuntimeError: Se o PDF não tiver texto extraível.
    """
    settings = get_settings()
    pages = load_pdf(settings.pdf_path)
    chunks = split_documents(pages)
    if not chunks:
        message = f"Nenhum texto extraído de '{settings.pdf_path}'."
        raise RuntimeError(message)

    store = get_vector_store(settings)
    store.add_documents(documents=chunks, ids=build_chunk_ids(chunks))

    print(f"PDF: {settings.pdf_path}")
    print(f"Páginas lidas: {len(pages)}")
    print(f"Chunks gravados: {len(chunks)}")
    return len(chunks)


if __name__ == "__main__":
    ingest_pdf()
