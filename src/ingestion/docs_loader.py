import os
from git import Repo
from langchain_text_splitters import MarkdownHeaderTextSplitter
from database.vector_store import VectorStore
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

from utils.logging_config import setup_logging

load_dotenv()

logger = setup_logging(__name__)


class DocsLoader:
    def __init__(self):
        self.repo_url = "https://github.com/tiangolo/fastapi"
        self.local_path = "./data/raw_docs/fastapi"
        self.vector_store = VectorStore()

    def clone_repo(self):
        if not os.path.exists(self.local_path):
            logger.info("Cloning docs repo", extra={"repo": self.repo_url})
            Repo.clone_from(self.repo_url, self.local_path)
        else:
            logger.info("Docs already exist locally")

    def load_and_split(self):
        # We define headers to split on so we keep context together
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        all_chunks = []

        # Walking through the FastAPI docs folder
        docs_dir = os.path.join(self.local_path, "docs", "en", "docs")
        if not os.path.exists(docs_dir):
            logger.warning("Docs directory not found. Cloning repo.")
            self.clone_repo()

        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".md"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks = splitter.split_text(content)
                        # Add metadata about the file source
                        for chunk in chunks:
                            chunk.metadata["source"] = file
                        all_chunks.extend(chunks)

        return all_chunks

    def upload_to_qdrant(self, chunks):
        logger.info("Uploading chunks to Qdrant", extra={"count": len(chunks)})

        # Use the client directly for in-memory operation
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Create collection if it doesn't exist
        try:
            self.vector_store.client.get_collection(self.vector_store.collection_name)
        except Exception:
            self.vector_store.client.create_collection(
                collection_name=self.vector_store.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        # Convert chunks to points
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
                    },
                )
            )

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.vector_store.client.upsert(
                collection_name=self.vector_store.collection_name, points=batch
            )

        logger.info("Documentation indexed successfully")


if __name__ == "__main__":
    loader = DocsLoader()
    chunks = loader.load_and_split()
    loader.upload_to_qdrant(chunks)
