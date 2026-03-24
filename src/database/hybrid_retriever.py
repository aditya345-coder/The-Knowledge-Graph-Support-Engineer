from database.vector_store import VectorStore
from database.graph_store import GraphStore

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()

    def retrieve_all(self, query: str, detected_feature: str = None):
        """Combines knowledge from both Vector and Graph databases."""
        docs = self.vector_store.search(query)
        bugs = []
        if detected_feature and detected_feature.lower() != "none":
            bugs = self.graph_store.get_related_issues(detected_feature)
            
        return {
            "official_docs": docs,
            "known_issues": bugs
        }
