import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_openai import ChatOpenAI
from agents.support_agent import SupportAgent

# 1. Load Dataset
with open("src/evaluation/golden_dataset.json", "r") as f:
    dataset = json.load(f)

# 2. Prepare Agent and Evaluator
agent = SupportAgent()
evaluator_llm = ChatOpenAI(model="gpt-4o") # RAGAS works best with GPT-4 class models

def run_evaluation():
    results = []
    for item in dataset:
        # Run agent
        state = agent.app.invoke({"query": item["user_query"]})
        
        # Format for RAGAS (expects context as a list of strings)
        results.append({
            "question": item["user_query"],
            "answer": state["response"],
            "contexts": state["documents"] + state["github_issues"],
            "ground_truth": item["ground_truth"]
        })

    # 3. Compute Metrics
    scores = evaluate(
        results,
        metrics=[faithfulness, answer_relevance, context_precision],
        llm=evaluator_llm
    )
    print(scores)

if __name__ == "__main__":
    run_evaluation()