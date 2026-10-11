
import re

from langchain_core.documents import Document


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in",
    "on", "at", "for", "with", "by", "from", "as", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "what", "which", "who", "how", "when", "where",
    "why", "do", "does", "did", "has", "have", "had",
}


class KeyWordRetriever:
    """Retrieve document chunks based on keyword overlap with the query."""

    def __init__(self, documents: list[Document]):
        self.update(documents)

    def update(self, documents: list[Document]) -> None:
        """Replace the searchable chunks after new uploads."""
        self.indexed = [
            (document, self._tokenize(document.page_content))
            for document in documents
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Return chunks ranked by the number of matching keywords."""
        query_words = self._tokenize(query)

        scored = []
        for document, document_words in self.indexed:
            score = len(query_words & document_words)

            if score > 0:
                scored.append((document, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Convert text into lowercase words and remove common stopwords."""
        words = set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))
        return words - STOPWORDS