import re
from langchain_core.documents import Document
def clean_documents(documents : list[Document]) -> list[Document]:
    """Clean WhiteSpace in document text while preserving meta data"""
    cleaned_documents=[]
    for document in documents:
        text=document.page_content
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        cleaned_documents.append(Document(
            page_content=text,
            metadata=document.metadata.copy(),
        ))
    return cleaned_documents
