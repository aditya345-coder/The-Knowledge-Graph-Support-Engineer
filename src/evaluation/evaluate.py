import json
import os
from dotenv import load_dotenv
from datasets import Dataset
import time # Add this import
from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
# from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
import litellm
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from agents.support_agent import SupportAgent

load_dotenv()

# 1. Load Dataset
with open("src/evaluation/golden_dataset.json", "r") as f:
    dataset = json.load(f)

# 2. Prepare Agent and Evaluator
agent = SupportAgent()

# Setup LiteLLM for Judge LLM
# evaluator_llm = ChatLiteLLM(
#     model=os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant")
# )

evaluator_llm = llm_factory(
    model=os.getenv("LLM_MODEL", "groq/llama-3.1-8b-instant"),
    provider="litellm",
    client=litellm.completion
)

# Setup local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def extract_text(docs):
    """Deep flatten + convert everything to string"""

    def flatten(item):
        if isinstance(item, list):
            result = []
            for sub in item:
                result.extend(flatten(sub))  # recursive flatten
            return result
        else:
            if hasattr(item, "page_content"):
                return [str(item.page_content)]
            return [str(item)]

    return flatten(docs)

def run_evaluation():
    results = []
    print("Running Agent through evaluation dataset...")

    for item in dataset:
        state = agent.app.invoke({"query": item["user_query"]})

        # Ensure we flatten everything into one single list of strings
        # state['documents'] and state['github_issues'] are already lists of strings
        # based on your support_agent.py code.
        time.sleep(2)
        flat_contexts = state.get("documents", []) + state.get("github_issues", [])

        results.append({
            "question": item["user_query"],
            "answer": state.get("response", ""),
            "retrieved_contexts": flat_contexts,  # Must be List[str]
            "ground_truth": item["ground_truth"]
        })

    # Create dataset
    eval_dataset = Dataset.from_list(results)

    # 3. Compute Metrics
    print("Computing metrics via RAGAS...")

    # Important: Initialize metric classes with ()
    # 1. Initialize the metrics with your chosen LLM and Embeddings
    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm)
    context_precision = ContextPrecision(llm=evaluator_llm)

    # 2. Pass the initialized objects to evaluate
    scores = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision], 
        llm=evaluator_llm,
        embeddings=embeddings
    )

    print(scores)
    
if __name__ == "__main__":
    run_evaluation()