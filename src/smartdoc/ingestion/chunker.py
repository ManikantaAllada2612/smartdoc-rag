from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks and give each chunk a unique ID.


    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for number, chunk in enumerate(chunks, start=1):
        source = chunk.metadata["source"]
        chunk.metadata["chunk_id"] = f"{source}-{number}"

    return chunks