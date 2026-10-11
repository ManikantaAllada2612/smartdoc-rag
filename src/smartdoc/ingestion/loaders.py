import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)


def load_document(file_path: str) -> list[Document]:
    """Load a PDF, TXT, or Markdown file into LangChain Documents.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: unsupported type, unreadable file, or no text in the file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        documents = _load_pdf(path)
    elif extension in {".txt", ".md"}:
        documents = _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    if not documents:
        raise ValueError(
            f"No extractable text found in {path.name}. "
            "It may be empty or a scanned PDF (images only)."
        )

    logger.info("Loaded %s: %d page(s)/section(s) with text", path.name, len(documents))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    """Read a PDF page by page, skipping pages that have no text."""
    try:
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError(f"{path.name} is password-protected and cannot be read.")

        pages = [page.extract_text() or "" for page in reader.pages]
    except PyPdfError as error:
        raise ValueError(f"Could not read PDF {path.name}: {error}") from error

    documents = []
    for page_number, text in enumerate(pages, start=1):
        if not text.strip():
            logger.warning("%s: page %d has no extractable text, skipped", path.name, page_number)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )
    return documents


def _load_text(path: Path) -> list[Document]:
    """Read a TXT/Markdown file as one Document (it has no pages)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8 text.") from error

    if not text.strip():
        return []

    return [Document(page_content=text, metadata={"source": path.name, "page": None})]