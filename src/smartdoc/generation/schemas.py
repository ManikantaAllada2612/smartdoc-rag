from pydantic import BaseModel, Field


class Source(BaseModel):
    """Where a piece of evidence came from."""

    document: str
    page: int | None
    chunk_id: str


class RAGResponse(BaseModel):
    """The final, validated answer returned to the user. Built by our code, not by the LLM."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source]
    grounded: bool


class LLMAnswer(BaseModel):
    """The only things we ask the LLM for: an answer and which chunks it used."""

    answer: str
    used_chunk_ids: list[str]


class RetrievedChunk(BaseModel):
    """One retrieved chunk, in a form that is easy to show, send as JSON, and evaluate."""

    chunk_id: str
    document: str
    page: int | None
    text: str
    score: float


class RAGResult(BaseModel):
    """Everything the pipeline produces for one question."""

    response: RAGResponse
    strategy: str
    retrieved_chunks: list[RetrievedChunk]
    latency_s: float