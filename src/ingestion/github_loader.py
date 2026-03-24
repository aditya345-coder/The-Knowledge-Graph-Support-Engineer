import os
import json
from github import Github
from database.graph_store import GraphStore
from agents.llm_gateway import LLMGateway
from dotenv import load_dotenv

from utils.logging_config import setup_logging

load_dotenv()

logger = setup_logging(__name__)


class GitHubGraphLoader:
    def __init__(self):
        self.gh = Github(os.getenv("GITHUB_TOKEN"))
        self.repo = self.gh.get_repo(os.getenv("TARGET_REPO"))
        self.graph_store = GraphStore()
        self.llm = LLMGateway()

    def extract_graph_data(self, issue_body):
        """Uses LLM to identify entities and relationships from an issue."""
        prompt = f"""
        Extract a Knowledge Graph from this GitHub Issue.
        Return ONLY valid JSON with keys: 'features', 'versions', 'relationships'.
        Issue: {issue_body[:2000]}
        """
        json_content = self.llm.extract_json(prompt)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")
            return {"features": [], "versions": [], "relationships": []}

    def save_to_neo4j(self, issue_id, graph_data):
        with self.graph_store.driver.session() as session:
            # Cypher query to create nodes and links
            query = """
            MERGE (i:Issue {neo4j_id: $id})
            SET i.title = $title
            FOREACH (feat IN $features | 
                MERGE (f:Feature {name: feat})
                MERGE (i)-[:AFFECTS]->(f))
            """
            session.run(query, id=str(issue_id), title=graph_data.get("title", "No Title"), features=graph_data.get('features', []))

    def run(self):
        # Fetching last 20 closed issues to build the graph
        issues = self.repo.get_issues(state="closed")[:20]
        for issue in issues:
            logger.info("Processing issue", extra={"issue": issue.number})
            data = self.extract_graph_data(issue.body)
            self.save_to_neo4j(issue.number, data)
        logger.info("Graph populated")


if __name__ == "__main__":
    loader = GitHubGraphLoader()
    loader.run()
