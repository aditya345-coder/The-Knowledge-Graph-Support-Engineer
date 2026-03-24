import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def get_related_issues(self, feature_name: str, limit: int = 5):
        """Finds issues affecting a specific feature."""
        query = """
        MATCH (f:Feature {name: $name})<-[:AFFECTS]-(i:Issue)
        RETURN i.id AS issue_id, i.title AS title LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, name=feature_name, limit=limit)
            return [f"Issue #{record['issue_id']}: {record['title']}" for record in result]
