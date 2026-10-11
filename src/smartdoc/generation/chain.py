
import logging
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ..config import LLM_MODEL
from .prompts import rag_prompt
from .schemas import LLMAnswer

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when the LLM cannot generate a valid answer."""


@lru_cache(maxsize=1)
def _get_chain():
    """Create and reuse the LLM generation chain."""

    load_dotenv()

    model = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
    )

    structured_model = model.with_structured_output(
        LLMAnswer,
        method="json_schema",
        strict=True,
    )

    return rag_prompt | structured_model


def generate_answer(question: str, context: str) -> LLMAnswer:
    """Generate an answer using the retrieved document context."""

    try:
        result = _get_chain().invoke({
            "question": question,
            "context": context,
        })

        if not result.answer.strip():
            raise ValueError("The LLM returned an empty answer.")

        return result

    except Exception as error:
        logger.exception("LLM generation failed")
        raise GenerationError(
            "Failed to generate an answer."
        ) from error
