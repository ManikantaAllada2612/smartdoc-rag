import logging

from .ingestion.ingest import ingest_document
from .retrieval.vector import VectorStore

logger = logging.getLogger(__name__)


def index_document(file_path: str, vector_store: VectorStore) -> int:
    """Ingest a document, store its chunks in the vector store, return the chunk count."""
    chunks = ingest_document(file_path)
    vector_store.add_documents(chunks)

    logger.info("Indexed %s (%d chunks)", file_path, len(chunks))
    return len(chunks)