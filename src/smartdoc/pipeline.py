import logging
import time

from langchain_core.documents import Document

from .config import DEFAULT_TOP_K
from .generation.chain import generate_answer
from .generation.context import build_context
from .generation.grounding import build_response
from .generation.schemas import RAGResponse, RAGResult, RetrievedChunk
from .retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

NO_ANSWER_TEXT = "I could not find relevant information in the provided documents."


class RAGPipeline:
    """Connect retrieval and generation into one RAG pipeline."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResult:
        """Retrieve relevant chunks, generate an answer, and check its grounding.

        Raises:
            ValueError: empty question or invalid top_k.
            GenerationError: the LLM failed or returned unusable output.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        start = time.perf_counter()

        results = self.retriever.search(question, top_k)
        chunks = [self._to_chunk(document, score) for document, score in results]

        if chunks:
            llm_answer = generate_answer(context=build_context(chunks), question=question)
            response = build_response(llm_answer, chunks)
        else:
            response = RAGResponse(answer=NO_ANSWER_TEXT, confidence=0.0, sources=[], grounded=False)

        latency = time.perf_counter() - start
        logger.info(
            "strategy=%s retrieved=%d cited=%d grounded=%s confidence=%.2f latency=%.2fs",
            self.retriever.strategy, len(chunks), len(response.sources),
            response.grounded, response.confidence, latency,
        )

        return RAGResult(
            response=response,
            strategy=self.retriever.strategy,
            retrieved_chunks=chunks,
            latency_s=latency,
        )

    @staticmethod
    def _to_chunk(document: Document, score: float) -> RetrievedChunk:
        """Convert a retrieved LangChain Document into a RetrievedChunk."""
        return RetrievedChunk(
            chunk_id=document.metadata["chunk_id"],
            document=document.metadata["source"],
            page=document.metadata.get("page"),
            text=document.page_content,
            score=float(score),
        )