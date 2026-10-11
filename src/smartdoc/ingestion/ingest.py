import logging

from langchain_core.documents import Document

from .chunker import chunk_documents
from .cleaner import clean_documents
from .loaders import load_document

logger = logging.getLogger(__name__)


def ingest_document(file_path: str) -> list[Document]:
    """Load, clean and chunk one document."""
    documents = load_document(file_path)
    cleaned_documents = clean_documents(documents)
    chunks = chunk_documents(cleaned_documents)

    logger.info("Ingested %s into %d chunk(s)", file_path, len(chunks))
    return chunks