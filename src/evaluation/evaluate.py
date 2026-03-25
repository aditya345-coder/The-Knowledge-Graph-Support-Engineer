import json
import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
# Use the correct v0.4.x import path
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from openai import OpenAI # Use the base OpenAI client for the wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from agents.support_agent import SupportAgent
from ragas.embeddings.base import LangchainEmbeddingsWrapper # Critical import

load_dotenv()

# 1. Load Dataset
with open("src/evaluation/golden_dataset.json", "r") as f:
    dataset = json.load(f)

# 2. Prepare Agent
agent = SupportAgent()

# 3. Setup NVIDIA NIM judge as a standard OpenAI-compatible client
# This is required by RAGAS 0.4+ llm_factory
raw_client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1" # Your NIM endpoint
)

# Use llm_factory to create the RAGAS-compatible LLM
evaluator_llm = llm_factory(
    model=os.getenv("LLM_MODEL", "meta/llama-3.1-405b-instruct"),
    provider="openai", 
    client=raw_client
)

# 4. Setup local embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Wrap the LangChain object for Ragas compatibility
embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

def run_evaluation():
    results = []
    print("Running Evaluation (v0.4.x)...")

    for item in dataset:
        state = agent.app.invoke({"query": item["user_query"]})
        
        # Flatten contexts
        flat_contexts = state.get("documents", []) + state.get("github_issues", [])

        results.append({
            "user_input": item["user_query"],         # formerly question
            "response": state.get("response", ""),    # formerly answer
            "retrieved_contexts": flat_contexts,
            "reference": item["ground_truth"]         # formerly ground_truth
        })

    eval_dataset = Dataset.from_list(results)

    # 5. Compute Metrics (v0.4.x requires initialized metric objects)
    f = Faithfulness(llm=evaluator_llm)
    ar = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings, strictness=1)
    cp = ContextPrecision(llm=evaluator_llm)


    # 3. Pass the initialized objects to evaluate
    scores = evaluate(
        dataset=eval_dataset,
        metrics=[f, ar, cp],
        llm=evaluator_llm,        # Global fallback LLM
        embeddings=embeddings     # Global fallback Embeddings
    )
    print(scores)

if __name__ == "__main__":
    run_evaluation()