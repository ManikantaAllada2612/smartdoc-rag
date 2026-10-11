import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class EvaluationQuestion(BaseModel):
    """One question used to evaluate the RAG system."""

    question: str
    expected_answers: list[str] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)

    @field_validator(
        "question",
        "expected_answers",
        "relevant_chunk_ids",
    )
    @classmethod
    def validate_not_blank(cls, value):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Value cannot be blank.")
        else:
            if any(not item.strip() for item in value):
                raise ValueError("Values cannot be blank.")

        return value


def load_evaluation_dataset(
    file_path: str | Path,
) -> list[EvaluationQuestion]:
    """Load and validate the evaluation dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in evaluation dataset: {path}"
        ) from error

    if not isinstance(data, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    questions = []

    for index, item in enumerate(data, start=1):
        try:
            questions.append(
                EvaluationQuestion(**item)
            )
        except Exception as error:
            raise ValueError(
                f"Question #{index} is invalid: {error}"
            ) from error

    if not questions:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    return questions


def validate_chunk_ids(
    dataset: list[EvaluationQuestion],
    available_chunk_ids: list[str],
) -> None:
    """Check that every labelled chunk exists in the vector store."""

    available = set(available_chunk_ids)

    missing = {
        chunk_id
        for item in dataset
        for chunk_id in item.relevant_chunk_ids
        if chunk_id not in available
    }

    if missing:
        raise ValueError(
            "Evaluation dataset contains unknown chunk IDs: "
            + ", ".join(sorted(missing))
        )