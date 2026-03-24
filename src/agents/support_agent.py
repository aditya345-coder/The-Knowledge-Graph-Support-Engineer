import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from database.hybrid_retriever import HybridRetriever
from agents.llm_gateway import LLMGateway

load_dotenv()

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
            "verify",
            self.should_continue,
            {
                "retry": "retrieve",
                "end": END
            }
        )
        
        self.app = self.workflow.compile()

    def analyze_query(self, state: AgentState):
        prompt = f"Identify the specific FastAPI feature in this query: '{state['query']}'. Return only the feature name or 'None'."
        res = self.llm.chat([{"role": "user", "content": prompt}])
        return {"detected_feature": res.choices[0].message.content.strip(), "iteration": 0}

    def retrieve_context(self, state: AgentState):
        context = self.retriever.retrieve_all(state['query'], state['detected_feature'])
        return {
            "documents": context['official_docs'], 
            "github_issues": context['known_issues'],
            "iteration": state.get("iteration", 0) + 1
        }

    def generate_answer(self, state: AgentState):
        context_str = f"DOCS: {state['documents']}\nISSUES: {state['github_issues']}"
        prompt = f"Use this context to solve: {state['query']}\nContext: {context_str}\nProvide a technical answer."
        res = self.llm.chat([{"role": "user", "content": prompt}])
        return {"response": res.choices[0].message.content}

    def verify_answer(self, state: AgentState):
        """The Critic node: checks if the answer is grounded in the provided documents."""
        context_str = f"DOCS: {state['documents']}\nISSUES: {state['github_issues']}"
        prompt = f"""
        Analyze if the following answer is grounded in the context provided.
        Answer: {state['response']}
        Context: {context_str}
        
        Does the answer contain information NOT present in the context? 
        Return ONLY 'True' if it is a hallucination, or 'False' if it is grounded.
        """
        res = self.llm.chat([{"role": "user", "content": prompt}])
        is_hallu = "true" in res.choices[0].message.content.lower()
        return {"is_hallucination": is_hallu}

    def should_continue(self, state: AgentState):
        if state["is_hallucination"] and state["iteration"] < 3:
            return "retry"
        return "end"
