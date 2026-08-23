import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, AnswerCorrectness
from ragas.llms import llm_factory
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import settings

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]

RESULTS_DIR = settings.PROJECT_ROOT / "data" / "eval_results"


def _extract_raw_text(formatted: str) -> str:
    match = re.match(r"^\[Source: .*?\] \| Content: (.*)", formatted)
    return match.group(1) if match else formatted


def _extract_raw_contexts(state: dict) -> list[str]:
    return [
        _extract_raw_text(d) for d in state.get("documents", [])
    ] + [
        _extract_raw_text(i) for i in state.get("github_issues", [])
    ]


def print_breakdown(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> None:
    for group_name, group_df in df.groupby(group_col, sort=False):
        n = len(group_df)
        print(f"\n  {group_name.upper()} (n={n}):")
        for m in metric_cols:
            mean_val = group_df[m].mean()
            std_val = group_df[m].std()
            print(f"    {m:25s}  {mean_val:.4f} ± {std_val:.4f}")


def save_results(df: pd.DataFrame, dataset: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    samples = []
    for i, row in df.iterrows():
        samples.append({
            "user_query": dataset[i]["user_query"],
            "difficulty": dataset[i]["difficulty"],
            "category": dataset[i]["category"],
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["context_precision"],
            "answer_correctness": row["answer_correctness"],
        })

    summary = {"timestamp": timestamp, "total_samples": len(df)}
    for m in METRIC_COLS:
        summary[f"overall_{m}_mean"] = float(df[m].mean())
        summary[f"overall_{m}_std"] = float(df[m].std())

    for diff_name, group_df in df.groupby("difficulty", sort=False):
        for m in METRIC_COLS:
            summary[f"{diff_name}_{m}_mean"] = float(group_df[m].mean())
            summary[f"{diff_name}_{m}_std"] = float(group_df[m].std())

    output = {"summary": summary, "samples": samples}

    filepath = results_dir / f"eval_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")

    latest_path = results_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({"latest": filepath.name}, f)


def evaluate_single(item: dict, repo_name: str, session_id: str, agent, force_relevant: bool = False) -> tuple[dict | None, dict | None]:
    q = item["user_query"]
    try:
        state = agent.app.invoke(
            {
                "query": q,
                "original_query": q,
                "rewritten_query": "",
                "repo_name": repo_name,
                "session_id": session_id,
                "detected_feature": "",
                "documents": [],
                "github_issues": [],
                "web_results": [],
                "response": "",
                "is_relevant": True,
                "is_hallucination": False,
                "iteration": 0,
                "allow_web_search": False,
            },
            config={"configurable": {"thread_id": f"eval-{session_id}-{hash(q) & 0xFFFFFFFF:08x}"}},
        )

        # Force retrieval even if relevancy gate rejected the query
        if force_relevant and not state.get("is_relevant"):
            state["is_relevant"] = True
            state.update(agent.rewrite_query(state))
            state.update(agent.retrieve_context(state))
            state.update(agent.generate_answer(state))
            state.update(agent.verify_answer(state))

        flat_contexts = _extract_raw_contexts(state)
        is_relevant = state.get("is_relevant")
        print(f"  RESPONSE: {state.get('response', '')[:200]}")
        print(f"  CONTEXTS: {len(flat_contexts)}, IS_RELEVANT: {is_relevant}")
        result = {
            "user_input": q,
            "response": state.get("response", ""),
            "retrieved_contexts": flat_contexts,
            "reference": item["ground_truth"],
        }
        return result, None
    except Exception as e:
        return None, {"user_query": q, "error": str(e)}


def run_evaluation(
    max_workers: int = 1,
    session_id: str = "",
    repo_name: str | None = None,
    force_relevant: bool = False,
    dataset_name: str = "generic",
) -> None:
    dataset_path = settings.PROJECT_ROOT / "src" / "evaluation" / "datasets" / f"{dataset_name}.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    from agents.support_agent import SupportAgent
    agent = SupportAgent()

    resolved_repo_name = repo_name or settings.TARGET_REPO or "this repository"

    raw_client = OpenAI(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1"
    )
    evaluator_llm = llm_factory(
        model=os.getenv("LLM_MODEL", "meta/llama-3.1-405b-instruct"),
        provider="openai",
        client=raw_client
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    results: list[dict[str, Any]] = []
    errors = []

    print(f"Evaluating {len(dataset)} questions...")

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_single, item, resolved_repo_name, session_id, agent, force_relevant): idx
                for idx, item in enumerate(dataset)
            }
            indexed_results: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                result, error = future.result()
                if error:
                    errors.append(error)
                    indexed_results[idx] = {
                        "user_input": dataset[idx]["user_query"],
                        "response": "",
                        "retrieved_contexts": [],
                        "reference": dataset[idx]["ground_truth"],
                    }
                    print(f" {idx + 1}/{len(dataset)} ✗ {dataset[idx]['user_query'][:60]}... ERROR: {error['error']}")
                elif result is not None:
                    indexed_results[idx] = result
                    print(f" {idx + 1}/{len(dataset)} OK {dataset[idx]['user_query'][:60]}...")
            for i in range(len(dataset)):
                if i in indexed_results:
                    results.append(indexed_results[i])
    else:
        for idx, item in enumerate(dataset):
            q = item["user_query"]
            result, error = evaluate_single(item, resolved_repo_name, session_id, agent, force_relevant=force_relevant)
            if error:
                errors.append(error)
                results.append({
                    "user_input": q,
                    "response": "",
                    "retrieved_contexts": [],
                    "reference": item["ground_truth"],
                })
                print(f" {idx + 1}/{len(dataset)} FAIL {q[:60]}... ERROR: {error['error']}")
            elif result is not None:
                results.append(result)
                print(f" {idx + 1}/{len(dataset)} OK {q[:60]}...")

    eval_dataset = Dataset.from_list(results)

    f = Faithfulness(llm=evaluator_llm)
    ar = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings, strictness=1)
    cp = ContextPrecision(llm=evaluator_llm)
    ac = AnswerCorrectness(llm=evaluator_llm, embeddings=embeddings)

    scores = evaluate(  # type: ignore[call-overload]
        dataset=eval_dataset,
        metrics=[f, ar, cp, ac],
        llm=evaluator_llm,  # type: ignore[arg-type]
        embeddings=embeddings
    )

    df = scores.to_pandas()  # type: ignore[union-attr]
    df["difficulty"] = [item["difficulty"] for item in dataset]
    df["category"] = [item["category"] for item in dataset]

    print("\n── Per Difficulty ──")
    print_breakdown(df, "difficulty", METRIC_COLS)

    print("\n── Per Category ──")
    print_breakdown(df, "category", METRIC_COLS)

    print("\n── Overall (All Questions) ──")
    print(f"{'Total samples:':30s} {len(df)}")
    for m in METRIC_COLS:
        mean_val = df[m].mean()
        std_val = df[m].std()
        print(f"  {m:30s}  {mean_val:.4f} ± {std_val:.4f}")

    if errors:
        print(f"\n── Errors ({len(errors)}/{len(dataset)}) ──")
        for e in errors:
            print(f"  FAIL {e['user_query'][:60]} -> {e['error']}")

    save_results(df, dataset, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Support Agent")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--session-id", type=str, default="",
        help="Target session for Qdrant/Neo4j queries (default: empty = docs_default, unfiltered Neo4j)")
    parser.add_argument("--repo", type=str, default=None,
        help="Repo name override for LLM prompts (default: settings.TARGET_REPO or 'this repository')")
    parser.add_argument("--force-relevant", action="store_true",
        help="Skip the relevancy gate — route all questions through retrieval (for evaluation only)")
    parser.add_argument("--dataset", type=str, default="generic",
        help="Dataset to evaluate (generic, flask, etc.). Loaded from datasets/<name>.json")
    args = parser.parse_args()

    run_evaluation(
        max_workers=args.workers,
        session_id=args.session_id,
        repo_name=args.repo,
        force_relevant=args.force_relevant,
        dataset_name=args.dataset,
    )
