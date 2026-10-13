
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from smartdoc.config import (
    DEFAULT_TOP_K,
    DOCUMENTS_DIR,
    FRONTEND_DIR,
    RESULTS_PATH,
)
from smartdoc.document_pipeline import index_document
from smartdoc.generation.chain import GenerationError
from smartdoc.pipeline import RAGPipeline
from smartdoc.retrieval.retriever import STRATEGIES, Retriever
from smartdoc.retrieval.vector import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartDoc RAG",
    description="An evaluated document question-answering system using RAG.",
    version="0.1.0",
)

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

vector_store = VectorStore()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: str = "hybrid"
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


@app.get("/api/health")
def health_check():
    """Check whether the API is running."""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF, TXT, or Markdown document."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and Markdown files are supported.",
        )

    destination = DOCUMENTS_DIR / filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A file with this name already exists. Rename the file and upload again.",
        )

    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)

        chunk_count = index_document(str(destination), vector_store)

        return {
            "message": "Document uploaded and indexed successfully.",
            "document": filename,
            "chunks_indexed": chunk_count,
        }

    except (FileNotFoundError, ValueError) as error:
        if destination.exists():
            destination.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

   
    except Exception as error:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                logger.exception(
                    "Could not remove failed upload: %s",
                    filename,
                )

        logger.exception("Document indexing failed: %s", filename)

        raise HTTPException(
            status_code=500,
            detail="The document could not be indexed.",
        ) from error


@app.post("/api/query")
def query_document(request: QueryRequest):
    """Ask a question using the selected retrieval strategy."""

    if request.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. "
                   f"Choose from {STRATEGIES}.",
        )

    try:
        retriever = Retriever(
            strategy=request.strategy,
            vector_store=vector_store,
        )

        pipeline = RAGPipeline(retriever)

        result = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
        )

        return result.model_dump()

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except GenerationError as error:
        logger.error("Generation failed: %s", error)

        raise HTTPException(
            status_code=502,
            detail="The language model could not generate an answer.",
        ) from error

    except Exception as error:
        logger.exception("Query failed.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the question.",
        ) from error


@app.get("/api/documents")
def list_documents():
    """List documents currently stored in the application."""

    documents = vector_store.get_all_documents()

    names = sorted(
        {
            document.metadata["source"]
            for document in documents
        }
    )

    return {
        "documents": names,
        "count": len(names),
    }


@app.get("/api/evaluation")
def get_evaluation():
    """Return previously generated evaluation results."""

    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results have not been generated yet.",
        )

    try:
        import json

        results = json.loads(
            RESULTS_PATH.read_text(encoding="utf-8")
        )

        return results

    except json.JSONDecodeError as error:
        logger.exception("Evaluation results are invalid JSON.")

        raise HTTPException(
            status_code=500,
            detail="Evaluation results are corrupted.",
        ) from error


if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )


@app.get("/")
def serve_frontend():
    """Serve the SmartDoc frontend."""

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend has not been created yet.",
        )

    return FileResponse(index_file)
