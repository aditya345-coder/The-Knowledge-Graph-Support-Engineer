import os
from typing import Annotated, TypedDict, List

from utils.logging_config import setup_logging
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.hybrid_retriever import HybridRetriever
from agents.llm_gateway import LLMGateway

from langgraph.pregel import Pregel

load_dotenv()

logger = setup_logging(__name__)


class AgentState(TypedDict):
    query: str
    detected_feature: str
    documents: List[str]
    github_issues: List[str]
    response: str
    is_hallucination: bool
    iteration: int


class SupportAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMGateway()
        self.workflow = StateGraph(AgentState)

        # Define nodes
        self.workflow.add_node("analyze", self.analyze_query)
        self.workflow.add_node("retrieve", self.retrieve_context)
        self.workflow.add_node("generate", self.generate_answer)
        self.workflow.add_node("verify", self.verify_answer)

        # Define flow
        self.workflow.set_entry_point("analyze")
        self.workflow.add_edge("analyze", "retrieve")
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.add_edge("generate", "verify")

        # Conditional Edge (The Loop)
        self.workflow.add_conditional_edges(
            "verify", self.should_continue, {"retry": "retrieve", "end": END}
        )

        self.app = self.workflow.compile()

        self.app.name = "FastAPI-Support-Agent"

    def analyze_query(self, state: AgentState):
        prompt = f"Identify the specific FastAPI feature in this query: '{state['query']}'. Return only the feature name or 'None'."
        res = self.llm.chat([{"role": "user", "content": prompt}])
        detected_feature = self.llm.get_message_text(res)
        logger.info("Detected feature", extra={"feature": detected_feature})
        return {"detected_feature": detected_feature, "iteration": 0}

    def retrieve_context(self, state: AgentState):
        context = self.retriever.retrieve_all(state["query"], state["detected_feature"])
        raw_docs = context.get("official_docs", [])
        formatted_docs = []
        for doc in raw_docs:
            if isinstance(doc, dict):
                source = doc.get("source", "unknown")
                text = doc.get("text", "")
                formatted_docs.append(f"[Source: {source}] | Content: {text}")
            else:
                formatted_docs.append(f"[Source: unknown] | Content: {doc}")

        raw_issues = context.get("known_issues", [])
        formatted_issues = []
        for issue in raw_issues:
            if isinstance(issue, str) and "Issue #" in issue and ":" in issue:
                prefix, title = issue.split(":", 1)
                issue_id = prefix.replace("Issue #", "").strip()
                formatted_issues.append(
                    f"[Source: Issue #{issue_id}] | Content: {title.strip()}"
                )
            else:
                formatted_issues.append(f"[Source: Issue #unknown] | Content: {issue}")
        logger.info(
            "Retrieved context",
            extra={
                "docs_count": len(formatted_docs),
                "issues_count": len(formatted_issues),
            },
        )
        return {
            "documents": formatted_docs,
            "github_issues": formatted_issues,
            "iteration": state.get("iteration", 0) + 1,
        }

    def generate_answer(self, state: AgentState):
        docs_context = "\n".join(state["documents"])
        issues_context = "\n".join(state["github_issues"])
        context_str = f"DOCS:\n{docs_context}\nISSUES:\n{issues_context}"
        prompt = (
            "You are a technical support assistant. For every factual claim, you must include the "
            "source tag provided in the context at the end of the sentence (e.g., [Source: docs.md]). "
            "If you do not have a source, do not state the fact. "
            "Use [Source: <filename>] for documents and [Source: Issue #<id>] for GitHub issues.\n"
            f"User question: {state['query']}\n"
            f"Context:\n{context_str}\n"
            "Provide a technical answer with citations."
        )
        res = self.llm.chat([{"role": "user", "content": prompt}])
        logger.info("Generated response")
        return {"response": self.llm.get_message_text(res)}

    def verify_answer(self, state: AgentState):
        """The Critic node: checks if the answer is grounded in the provided documents."""
        docs_context = "\n".join(state["documents"])
        issues_context = "\n".join(state["github_issues"])
        context_str = f"DOCS:\n{docs_context}\nISSUES:\n{issues_context}"
        prompt = f"""
        Analyze if the following answer is grounded in the context provided.
        Answer: {state["response"]}
        Context: {context_str}
        
        Does the answer contain information NOT present in the context? 
        Return ONLY 'True' if it is a hallucination, or 'False' if it is grounded.
        """
        res = self.llm.chat([{"role": "user", "content": prompt}])
        response_text = self.llm.get_message_text(res)
        is_hallu = "true" in response_text.lower()
        logger.info("Verification result", extra={"is_hallucination": is_hallu})
        return {"is_hallucination": is_hallu}

    def should_continue(self, state: AgentState):
        if state["is_hallucination"] and state["iteration"] < 3:
            return "retry"
        return "end"
