from langchain_core.documents import Document

from .keyword import KeyWordRetriever
from .vector import VectorStore


class HybridRetriever:
    """Combine vector and keyword scores into one ranking."""

    def __init__(self, vector_store: VectorStore, keyword_retriever: KeyWordRetriever):
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever

    @staticmethod
    def normalize_scores(scores: list[float]) -> list[float]:
        """Min-max normalize scores to the 0..1 range."""
        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0 for _ in scores]

        return [(score - minimum) / (maximum - minimum) for score in scores]

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> list[tuple[Document, float]]:
        """Return the top_k (chunk, combined score) pairs."""
        vector_results = self.vector_store.search(query, top_k)
        keyword_results = self.keyword_retriever.search(query, top_k)

        # Vector similarity and keyword counts are on different scales,
        # so normalize each list before combining.
        vector_scores = self.normalize_scores([score for _, score in vector_results])
        keyword_scores = self.normalize_scores([score for _, score in keyword_results])

        documents: dict[str, Document] = {}
        combined_scores: dict[str, float] = {}

        for (document, _), score in zip(vector_results, vector_scores):
            chunk_id = document.metadata["chunk_id"]
            documents[chunk_id] = document
            combined_scores[chunk_id] = vector_weight * score

        for (document, _), score in zip(keyword_results, keyword_scores):
            chunk_id = document.metadata["chunk_id"]
            documents[chunk_id] = document
            # A chunk found by both retrievers gets both contributions added.
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + keyword_weight * score

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        return [(documents[chunk_id], score) for chunk_id, score in ranked[:top_k]]