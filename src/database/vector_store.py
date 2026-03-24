import os
from typing import Any, Iterable, cast

from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

from utils.logging_config import setup_logging

load_dotenv()

logger = setup_logging(__name__)


class VectorStore:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Use in-memory Qdrant if no URL is provided
        if not qdrant_url or qdrant_url == "http://localhost:6333":
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant client")
        else:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info("Using Qdrant client", extra={"url": qdrant_url})
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.collection_name = "fastapi_docs"

    def search(self, query: str, limit: int = 3):
        """Performs semantic search on the documentation."""
        query_vector = self.embeddings.embed_query(query)
        client: Any = self.client
        search_fn = getattr(client, "search", None)
        if callable(search_fn):
            raw_points = search_fn(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            logger.debug("Qdrant search used")
            points = list(cast(Iterable[Any], raw_points))
        else:
            response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            logger.debug("Qdrant query_points used")
            raw_points = getattr(response, "points", [])
            points = list(cast(Iterable[Any], raw_points))
        results = []
        for res in points:
            payload = getattr(res, "payload", None) or {}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "feature_name": payload.get("feature_name")
                    or payload.get("feature")
                    or "General",
                    "neo4j_id": payload.get("neo4j_id")
                    or payload.get("feature_name")
                    or payload.get("feature")
                    or "General",
                    "score": getattr(res, "score", None),
                }
            )
        return results
