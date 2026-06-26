import json
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm_client import LLMClient
from judges.evaluators import KeywordJudge, LLMJudge, LatencyJudge
from reports.report_generator import ReportGenerator


def load_dataset(path: str = "data/golden_dataset.json") -> list:
    with open(path) as f:
        return json.load(f)


def run_eval(test_case: dict, client: LLMClient, keyword_judge: KeywordJudge,
             llm_judge: LLMJudge, latency_judge: LatencyJudge) -> dict:
    """Run a single test case and return result."""
    prompt = test_case["prompt"]
    runs = test_case.get("runs", 1)
    criteria = test_case.get("pass_criteria")

    all_responses = []
    all_latencies = []

    for _ in range(runs):
        llm_result = client.call(prompt)
        if llm_result["error"]:
            return {
                **test_case,
                "response": "",
                "latency_ms": llm_result["latency_ms"],
                "passed": False,
                "judge_results": [{"judge": "error", "passed": False, "score": 0.0,
                                   "detail": llm_result["error"]}],
            }
        all_responses.append(llm_result["response"])
        all_latencies.append(llm_result["latency_ms"])

    # Use first response for evaluation, average latency
    response = all_responses[0]
    avg_latency = sum(all_latencies) // len(all_latencies)

    judge_results = []

    # Consistency check for multiple runs
    if runs > 1:
        # Check all responses contain the same expected keywords
        keyword_j = keyword_judge
        consistent = all(
            any(kw.lower() in r.lower() for kw in test_case.get("expected_keywords", []))
            for r in all_responses
        ) if test_case.get("expected_keywords") else True
        judge_results.append({
            "judge": "consistency",
            "passed": consistent,
            "score": 1.0 if consistent else 0.0,
            "detail": f"Consistent across {runs} runs" if consistent else f"Inconsistent across {runs} runs",
        })

    # Primary judge based on criteria
    if criteria == "llm_judge":
        judge_results.append(llm_judge.evaluate(prompt, response, test_case))
    else:
        judge_results.append(keyword_judge.evaluate(response, test_case))

    # Latency judge (always runs if max_latency_ms set)
    latency_result = latency_judge.evaluate(avg_latency, test_case)
    if test_case.get("max_latency_ms"):
        judge_results.append(latency_result)

    # Overall pass = all judges pass
    overall_passed = all(j["passed"] for j in judge_results)
    overall_score = sum(j["score"] for j in judge_results) / len(judge_results)

    return {
        **test_case,
        "response": response,
        "all_responses": all_responses if runs > 1 else None,
        "latency_ms": avg_latency,
        "passed": overall_passed,
        "score": overall_score,
        "judge_results": judge_results,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Eval Framework")
    parser.add_argument("--dataset", default="data/golden_dataset.json", help="Path to golden dataset")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as new baseline")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare against saved baseline")
    parser.add_argument("--critical-only", action="store_true", help="Run only critical test cases")
    args = parser.parse_args()

    print("\n🔍 LLM Eval Framework Starting...\n")

    client = LLMClient()
    keyword_judge = KeywordJudge()
    llm_judge = LLMJudge()
    latency_judge = LatencyJudge()
    reporter = ReportGenerator()

    dataset = load_dataset(args.dataset)

    # Apply filters
    if args.category:
        dataset = [tc for tc in dataset if tc.get("category") == args.category]
    if args.critical_only:
        dataset = [tc for tc in dataset if tc.get("critical")]

    print(f"Running {len(dataset)} test cases against {client.model}\n")

    results = []
    for tc in dataset:
        print(f"  Running [{tc['id']}]...", end="", flush=True)
        result = run_eval(tc, client, keyword_judge, llm_judge, latency_judge)
        results.append(result)
        status = "✓" if result["passed"] else "✗"
        print(f" {status}")

    # Print detailed results
    print("\n--- DETAILED RESULTS ---")
    for result in results:
        reporter.print_result(result)

    # Summary
    reporter.print_summary(results)

    # Baseline handling
    baseline = None
    if args.compare_baseline:
        baseline = reporter.load_baseline()
        if baseline:
            print("Comparing against baseline...")
        else:
            print("No baseline found — run with --save-baseline first")

    report_file = reporter.save_json(results, baseline)

    if args.save_baseline:
        reporter.save_baseline(results)

    # Exit with error code if any critical tests fail
    critical_failures = [r for r in results if not r["passed"] and r.get("critical")]
    if critical_failures:
        print(f"❌ {len(critical_failures)} critical test(s) failed")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
