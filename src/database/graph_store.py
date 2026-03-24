import os
from typing import cast

from neo4j import GraphDatabase
from dotenv import load_dotenv

from utils.logging_config import setup_logging

load_dotenv()

logger = setup_logging(__name__)


class GraphStore:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not username or not password:
            logger.error("Missing Neo4j configuration")
            raise ValueError("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD must be set")
        uri = cast(str, uri)
        username = cast(str, username)
        password = cast(str, password)
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )
        logger.info("Neo4j driver initialized", extra={"uri": uri})

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
            issues = [
                f"Issue #{record['issue_id']}: {record['title']}" for record in result
            ]
            logger.info(
                "Related issues fetched",
                extra={"feature": feature_name, "count": len(issues)},
            )
            return issues
