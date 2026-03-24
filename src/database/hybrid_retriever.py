from database.vector_store import VectorStore
from database.graph_store import GraphStore
from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

    def retrieve_all(self, query: str, detected_feature: str | None = None):
        """Combines knowledge from both Vector and Graph databases."""
        docs = self.vector_store.search(query)
        bugs = []
        if isinstance(detected_feature, str) and detected_feature.lower() != "none":
            bugs = self.graph_store.get_related_issues(detected_feature)
        logger.info(
            "Hybrid retrieval complete",
            extra={"docs_count": len(docs), "issues_count": len(bugs)},
        )

        return {"official_docs": docs, "known_issues": bugs}
