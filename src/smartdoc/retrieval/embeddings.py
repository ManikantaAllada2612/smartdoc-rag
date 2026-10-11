from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL
class EmbeddingModel:
    """Create embeddings for document chunks and queries"""
    def __init__(self,model_name : str = EMBEDDING_MODEL):
        self.model=SentenceTransformer(model_name)
    def embed_text(self,text:str) -> list[float]:
        """Convert text into an embedding vector"""
        return self.model.encode(text).tolist()
    def embed_documents(self,texts : list[str]) -> list[list[float]] :
        """Convert mutiple texts into embedding vector"""
        return self.model.encode(texts).tolist()