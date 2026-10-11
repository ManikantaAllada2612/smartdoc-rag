"""Project configuration settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 800

CHUNK_OVERLAP = 120

DEFAULT_TOP_K = 5

VECTOR_STORE_PATH = str(PROJECT_ROOT / "data" / "chroma")

COLLECTION_NAME = "smartdoc"

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

QUESTIONS_PATH = EVALUATION_DIR / "questions.json"

RESULTS_PATH = EVALUATION_DIR / "results.json"

FRONTEND_DIR = PROJECT_ROOT / "frontend"