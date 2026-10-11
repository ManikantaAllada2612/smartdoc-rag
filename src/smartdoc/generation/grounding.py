
import logging
import re

from .schemas import LLMAnswer, RAGResponse, RetrievedChunk, Source

logger = logging.getLogger(__name__)

GROUNDED_THRESHOLD = 0.6

NO_ANSWER_TEXT = (
    "I could not find relevant information in the provided documents."
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on",
    "at", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "not", "no", "can", "will", "may", "per", "than",
    "then", "there", "their", "they", "you", "your", "we", "our",
    "i", "he", "she", "his", "her", "which", "who", "if", "do",
    "does", "has", "have", "had",
}


def _content_words(text: str) -> set[str]:
    """Return lowercase words after removing common stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {word for word in words if word not in STOPWORDS}


def support_ratio(answer: str, evidence_texts: list[str]) -> float:
    """Calculate how much of the answer's vocabulary appears in the evidence."""
    answer_words = _content_words(answer)

    if not answer_words:
        return 0.0

    evidence_words = set()
    for text in evidence_texts:
        evidence_words.update(_content_words(text))

    return len(answer_words & evidence_words) / len(answer_words)


def build_response(
    llm_answer: LLMAnswer,
    chunks: list[RetrievedChunk],
) -> RAGResponse:
    """Validate citations and calculate answer confidence."""

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    cited_ids = list(dict.fromkeys(llm_answer.used_chunk_ids))
    valid_ids = [
        chunk_id for chunk_id in cited_ids
        if chunk_id in chunks_by_id
    ]

    invalid_ids = [
        chunk_id for chunk_id in cited_ids
        if chunk_id not in chunks_by_id
    ]

    if invalid_ids:
        logger.warning("LLM cited unknown chunk IDs: %s", invalid_ids)

    if not valid_ids:
        return RAGResponse(
            answer=NO_ANSWER_TEXT,
            confidence=0.0,
            sources=[],
            grounded=False,
        )

    sources = [
        Source(
            document=chunks_by_id[chunk_id].document,
            page=chunks_by_id[chunk_id].page,
            chunk_id=chunk_id,
        )
        for chunk_id in valid_ids
    ]

    evidence_texts = [
        chunks_by_id[chunk_id].text for chunk_id in valid_ids
    ]

    support = support_ratio(llm_answer.answer, evidence_texts)
    citation_validity = len(valid_ids) / len(cited_ids)

    return RAGResponse(
        answer=llm_answer.answer,
        confidence=round(support * citation_validity, 2),
        sources=sources,
        grounded=support >= GROUNDED_THRESHOLD,
    )
