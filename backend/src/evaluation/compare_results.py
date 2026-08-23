"""Compare two most recent evaluation result JSONs.

Prints a side-by-side comparison with colored indicators.
Exits with non-zero code on regression.

Usage:
    python -m evaluation.compare_results
    python -m evaluation.compare_results --before eval_2026-06-20.json --after eval_2026-06-21.json
"""

import json
import sys
import argparse
from pathlib import Path

from settings import settings

RESULTS_DIR = settings.PROJECT_ROOT / "data" / "eval_results"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]

# Regression thresholds
WARNING_THRESHOLD = 0.02
BLOCKING_THRESHOLD = 0.05


def load_results(path: Path) -> dict:
    """Load an evaluation result JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metric(mean: float, std: float) -> str:
    """Format a metric as 'mean ± std'."""
    return f"{mean:.4f} ± {std:.4f}"


def compare(before: dict, after: dict) -> int:
    """Compare two evaluation results and print a side-by-side table.

    Returns:
        0: No regression
        1: Warning regression (drop > 0.02)
        2: Blocking regression (drop > 0.05)
    """
    before_summary = before.get("summary", {})
    after_summary = after.get("summary", {})

    before_ts = before_summary.get("timestamp", "unknown")
    after_ts = after_summary.get("timestamp", "unknown")

    print(f"\nComparison: {before_ts}  vs  {after_ts}")
    print(f"Timestamp:  {before_ts}  →  {after_ts}")

    max_regression = 0.0

    # Overall comparison
    print("\n── Overall ──")
    print(f"{'':30s} {'Before':>20s}  {'After':>20s}  {'Change':>10s}")
    print("-" * 85)

    for m in METRIC_COLS:
        before_mean = before_summary.get(f"overall_{m}_mean", 0.0)
        before_std = before_summary.get(f"overall_{m}_std", 0.0)
        after_mean = after_summary.get(f"overall_{m}_mean", 0.0)
        after_std = after_summary.get(f"overall_{m}_std", 0.0)

        change = after_mean - before_mean
        max_regression = min(max_regression, change)

        if change > WARNING_THRESHOLD:
            indicator = "🟢"
        elif change < -BLOCKING_THRESHOLD:
            indicator = "🔴"
        elif change < -WARNING_THRESHOLD:
            indicator = "🟡"
        else:
            indicator = "🟢"

        change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
        print(f"  {m:28s} {format_metric(before_mean, before_std):>20s}  {format_metric(after_mean, after_std):>20s}  {change_str:>9s} {indicator}")

    # Per-difficulty comparison
    difficulties = set()
    for key in before_summary:
        if "_faithfulness_mean" in key and "overall" not in key:
            diff = key.split("_faithfulness_mean")[0]
            difficulties.add(diff)

    if difficulties:
        print("\n── Per Difficulty ──")
        for diff in sorted(difficulties):
            print(f"\n  {diff.upper()}:")
            for m in METRIC_COLS:
                before_mean = before_summary.get(f"{diff}_{m}_mean", 0.0)
                before_std = before_summary.get(f"{diff}_{m}_std", 0.0)
                after_mean = after_summary.get(f"{diff}_{m}_mean", 0.0)
                after_std = after_summary.get(f"{diff}_{m}_std", 0.0)

                change = after_mean - before_mean
                max_regression = min(max_regression, change)

                if change > WARNING_THRESHOLD:
                    indicator = "🟢"
                elif change < -BLOCKING_THRESHOLD:
                    indicator = "🔴"
                elif change < -WARNING_THRESHOLD:
                    indicator = "🟡"
                else:
                    indicator = "🟢"

                change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
                print(f"    {m:26s} {format_metric(before_mean, before_std):>20s}  {format_metric(after_mean, after_std):>20s}  {change_str:>9s} {indicator}")

    # Determine exit code
    if max_regression < -BLOCKING_THRESHOLD:
        print(f"\n🔴 BLOCKING REGRESSION DETECTED (drop > {BLOCKING_THRESHOLD})")
        print("Deploy should be blocked until regression is addressed.")
        return 2
    elif max_regression < -WARNING_THRESHOLD:
        print(f"\n🟡 WARNING: Regression detected (drop > {WARNING_THRESHOLD})")
        print("Review the changes before deploying.")
        return 1
    else:
        print("\n🟢 No regression detected. Safe to deploy.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--before", type=str, default=None,
        help="Path to the 'before' result JSON (default: second most recent)")
    parser.add_argument("--after", type=str, default=None,
        help="Path to the 'after' result JSON (default: most recent)")
    args = parser.parse_args()

    if args.before and args.after:
        before_path = Path(args.before)
        after_path = Path(args.after)
    else:
        files = sorted(RESULTS_DIR.glob("eval_*.json"))
        if len(files) < 2:
            print("Need at least 2 result files to compare.")
            print(f"Found {len(files)} files in {RESULTS_DIR}")
            return 1
        before_path = files[-2]
        after_path = files[-1]

    if not before_path.exists():
        print(f"Error: {before_path} does not exist")
        return 1
    if not after_path.exists():
        print(f"Error: {after_path} does not exist")
        return 1

    before = load_results(before_path)
    after = load_results(after_path)

    return compare(before, after)


if __name__ == "__main__":
    sys.exit(main())
