from .schemas import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the context the LLM reads.

    Every chunk starts with a "[chunk_id: ...]" label, which the LLM must copy
    into used_chunk_ids so its citations can be checked.
    """
    parts = []

    for chunk in chunks:
        location = f"Document: {chunk.document}"
        if chunk.page is not None:
            location += f" | Page: {chunk.page}"

        parts.append(f"[chunk_id: {chunk.chunk_id}]\n{location}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)