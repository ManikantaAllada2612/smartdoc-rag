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
    """Measure how many of the top-k retrieved chunks are relevant."""

    retrieved = retrieved_chunk_ids[:k]

    if not retrieved or k <= 0:
        return 0.0

    relevant = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved
    )

    return relevant / k


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


def _tokenize(text: str) -> set[str]:
    """Convert text into lowercase unique words."""

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )


def token_f1(
    generated_answer: str,
    expected_answers: list[str],
) -> float:
    """Calculate the best token-level F1 against the reference answers."""

    generated_words = _tokenize(generated_answer)

    if not generated_words or not expected_answers:
        return 0.0

    best_score = 0.0

    for expected_answer in expected_answers:

        expected_words = _tokenize(expected_answer)

        if not expected_words:
            continue

        common_words = generated_words & expected_words

        precision = len(common_words) / len(generated_words)
        recall = len(common_words) / len(expected_words)

        if precision + recall == 0:
            score = 0.0
        else:
            score = (
                2 * precision * recall
                / (precision + recall)
            )

        best_score = max(best_score, score)

    return best_score