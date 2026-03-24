# src/ingestion/docs_loader.py
import os
from git import Repo
from langchain_text_splitters import MarkdownHeaderTextSplitter
from database.vector_store import VectorStore
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
from utils.logging_config import setup_logging

load_dotenv()
logger = setup_logging(__name__)


class DocsLoader:
    def __init__(self):
        self.repo_url = "https://github.com/tiangolo/fastapi"
        self.local_path = "./data/raw_docs/fastapi"
        self.vector_store = VectorStore()

    def identify_feature(self, text: str) -> str:
        """Simple keyword-based detector to bridge docs to graph features."""
        text = text.lower()
        if "backgroundtask" in text or "background tasks" in text:
            return "BackgroundTasks"
        if "dependency" in text or "depends(" in text:
            return "Dependency Injection"
        if "security" in text or "oauth" in text or "bearer" in text:
            return "Security"
        if "orm" in text or "sql" in text or "sqlalchemy" in text:
            return "SQLAlchemy/ORM"
        return "General"

    def resolve_neo4j_id(self, feature_name: str) -> str:
        """Maps a feature name to its Neo4j identifier."""
        return feature_name

    def load_and_split(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        all_chunks = []
        docs_dir = os.path.join(self.local_path, "docs", "en", "docs")

        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = splitter.split_text(content)
                        for chunk in chunks:
                            feature_name = self.identify_feature(chunk.page_content)
                            neo4j_id = self.resolve_neo4j_id(feature_name)
                            chunk.metadata["source"] = file
                            chunk.metadata["feature_name"] = feature_name
                            chunk.metadata["feature"] = feature_name
                            chunk.metadata["neo4j_id"] = neo4j_id
                        all_chunks.extend(chunks)
        return all_chunks

    def upload_to_qdrant(self, chunks):
        try:
            self.vector_store.client.get_collection(self.vector_store.collection_name)
        except Exception:
            self.vector_store.client.create_collection(
                collection_name=self.vector_store.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        points = []
        for i, chunk in enumerate(chunks):
            vector = self.vector_store.embeddings.embed_documents([chunk.page_content])[
                0
            ]
            points.append(
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "text": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "feature_name": chunk.metadata.get("feature_name", "General"),
                        "feature": chunk.metadata.get("feature", "General"),
                        "neo4j_id": chunk.metadata.get(
                            "neo4j_id",
                            chunk.metadata.get("feature_name", "General"),
                        ),
                    },
                )
            )
        self.vector_store.client.upsert(
            collection_name=self.vector_store.collection_name, points=points
        )
        logger.info("Documentation indexed with feature tags")


if __name__ == "__main__":
    loader = DocsLoader()
    chunks = loader.load_and_split()
    loader.upload_to_qdrant(chunks)
