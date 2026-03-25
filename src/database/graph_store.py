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

    # def get_related_issues(self, neo4j_id: str, limit: int = 5):
    #     """Finds issues affecting a specific feature."""
    #     query = """
    #     MATCH (f:Feature)
    #     WHERE f.neo4j_id = $neo4j_id OR f.name = $neo4j_id
    #     MATCH (f)<-[:AFFECTS]-(i:Issue)
    #     RETURN i.id AS issue_id, i.title AS title LIMIT $limit
    #     """
    #     with self.driver.session() as session:
    #         result = session.run(query, neo4j_id=neo4j_id, limit=limit)
    #         issues = [
    #             f"Issue #{record['issue_id']}: {record['title']}" for record in result
    #         ]
    #         logger.info(
    #             "Related issues fetched",
    #             extra={"feature": neo4j_id, "count": len(issues)},
    #         )
    #         return issues
    def get_related_issues(self, neo4j_id: str, limit: int = 5):
        query = """
        MATCH (f:Feature)
        WHERE f.neo4j_id = $neo4j_id OR f.name = $neo4j_id
        MATCH (f)<-[:AFFECTS]-(i:Issue)
        RETURN i.id AS issue_id, i.title AS title LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, neo4j_id=neo4j_id, limit=limit)
                return [f"Issue #{record['issue_id']}: {record['title']}" for record in result]
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            # Try to re-initialize the driver if it's a connection error
            try:
                self.driver.verify_connectivity()
            except:
                logger.warning("Neo4j connection lost. Attempting to reconnect...")
                # Add logic here to re-run __init__ if necessary
            return [] # Return empty list so the Agent doesn't crash
