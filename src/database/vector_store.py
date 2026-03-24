import os
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Use in-memory Qdrant if no URL is provided
        if not qdrant_url or qdrant_url == "http://localhost:6333":
            from qdrant_client import QdrantClient

            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.collection_name = "fastapi_docs"

    def search(self, query: str, limit: int = 3):
        """Performs semantic search on the documentation."""
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
        return [res.payload["text"] for res in results]
