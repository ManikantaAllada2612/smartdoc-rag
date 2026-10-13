import re


def hit_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """Return 1 if at least one relevant chunk is in the top-k."""

    retrieved = retrieved_chunk_ids[:k]

    return float(
        any(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved
        )
    )



def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """Measure the proportion of retrieved top-k chunks that are relevant."""

    if k <= 0:
        return 0.0

    retrieved = retrieved_chunk_ids[:k]

    if not retrieved:
        return 0.0

    relevant = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved
    )

    return relevant / len(retrieved)



def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """Measure how many relevant chunks were retrieved in the top-k."""

    if not relevant_chunk_ids or k <= 0:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])

    relevant_found = sum(
        chunk_id in retrieved
        for chunk_id in relevant_chunk_ids
    )

    return relevant_found / len(relevant_chunk_ids)



