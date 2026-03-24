from unittest.mock import patch

from database.hybrid_retriever import HybridRetriever


def test_retriever_uses_neo4j_id_from_docs():
    with (
        patch("database.hybrid_retriever.VectorStore") as mock_vector_store,
        patch("database.hybrid_retriever.GraphStore") as mock_graph_store,
    ):
        mock_vector_store.return_value.search.return_value = [
            {"neo4j_id": "Feature123"}
        ]

        retriever = HybridRetriever()
        retriever.retrieve_all("example query")

        mock_graph_store.return_value.get_related_issues.assert_called_once_with(
            "Feature123"
        )
