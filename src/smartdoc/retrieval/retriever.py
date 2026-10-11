from langchain_core.documents import Document

from .hybrid import HybridRetriever
from .keyword import KeyWordRetriever
from .vector import VectorStore

STRATEGIES = ("vector", "keyword", "hybrid")


class Retriever:
    """One interface for the three retrieval strategies."""

    def __init__(self, strategy: str, vector_store: VectorStore):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy: {strategy}. Choose from {STRATEGIES}")

        self.strategy = strategy
        self.vector_store = vector_store
        # The keyword index lives in memory, so rebuild it from Chroma (survives restarts).
        self.keyword_retriever = KeyWordRetriever(vector_store.get_all_documents())
        # Hybrid shares the same keyword retriever instead of building its own.
        self.hybrid_retriever = HybridRetriever(vector_store, self.keyword_retriever)

    def refresh(self) -> None:
        """Reload the keyword index from Chroma. Call after indexing a document."""
        self.keyword_retriever.update(self.vector_store.get_all_documents())

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """Return (chunk, score) pairs using the selected strategy."""
        if self.strategy == "vector":
            return self.vector_store.search(query, top_k)
        if self.strategy == "keyword":
            return self.keyword_retriever.search(query, top_k)
        return self.hybrid_retriever.search(query, top_k)