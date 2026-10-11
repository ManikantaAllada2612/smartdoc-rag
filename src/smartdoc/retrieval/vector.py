import logging

import chromadb
from langchain_core.documents import Document

from .embeddings import EmbeddingModel
from ..config import COLLECTION_NAME, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

# Chroma metadata cannot store None, so a missing page (TXT/MD files) is stored as -1.
NO_PAGE = -1


class VectorStore:
    """Store document chunks with their embeddings in Chroma and search them."""

    def __init__(
        self,
        persist_directory: str = VECTOR_STORE_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

        self.embedding_model = embedding_model or EmbeddingModel()

    def add_documents(self, documents: list[Document]) -> None:
        """Index chunks. Re-indexing a file replaces its old chunks."""
        if not documents:
            return

        # If a file is uploaded again, delete its old chunks first so
        # nothing is duplicated and no stale chunks stay behind.
        for source in {document.metadata["source"] for document in documents}:
            self.collection.delete(where={"source": source})

        texts = [document.page_content for document in documents]

        self.collection.add(
            ids=[document.metadata["chunk_id"] for document in documents],
            documents=texts,
            embeddings=self.embedding_model.embed_documents(texts),
            metadatas=[self._to_chroma_metadata(document.metadata) for document in documents],
        )
        logger.info("Indexed %d chunk(s); collection now has %d", len(documents), self.collection.count())

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """Return up to top_k (chunk, similarity) pairs, most similar first."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[self.embedding_model.embed_text(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            document = Document(page_content=text, metadata=self._from_chroma_metadata(metadata))
            retrieved.append((document, 1 - distance))  # cosine distance -> similarity
        return retrieved

    def get_all_documents(self) -> list[Document]:
        """Return every stored chunk (used to rebuild the keyword index)."""
        stored = self.collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=self._from_chroma_metadata(metadata))
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]

    @staticmethod
    def _to_chroma_metadata(metadata: dict) -> dict:
        saved = dict(metadata)
        if saved.get("page") is None:
            saved["page"] = NO_PAGE
        return saved

    @staticmethod
    def _from_chroma_metadata(metadata: dict) -> dict:
        restored = dict(metadata)
        if restored.get("page") == NO_PAGE:
            restored["page"] = None
        return restored