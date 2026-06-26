import json
import os
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)


class ReportGenerator:
    """Generates console and JSON reports from eval results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def print_result(self, result: dict):
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result["passed"] else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        critical = f"{Fore.YELLOW}[CRITICAL]{Style.RESET_ALL} " if result.get("critical") else ""
        print(f"\n{critical}[{result['id']}] {result['category'].upper()} — {status}")
        print(f"  Prompt: {result['prompt'][:80]}{'...' if len(result['prompt']) > 80 else ''}")
        print(f"  Response: {result['response'][:100]}{'...' if len(result.get('response','')) > 100 else ''}")
        print(f"  Latency: {result.get('latency_ms', 0)}ms")
        for judge_result in result.get("judge_results", []):
            j_status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if judge_result["passed"] else f"{Fore.RED}✗{Style.RESET_ALL}"
            print(f"  [{judge_result['judge']}] {j_status} Score: {judge_result['score']:.2f} — {judge_result['detail']}")

    def print_summary(self, results: list):
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed
        critical_failures = [r for r in results if not r["passed"] and r.get("critical")]

        print(f"\n{'='*60}")
        print(f"  LLM EVAL RESULTS")
        print(f"{'='*60}")
        print(f"  Total:   {total}")
        print(f"  {Fore.GREEN}Passed:  {passed}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Failed:  {failed}{Style.RESET_ALL}")
        if critical_failures:
            print(f"  {Fore.YELLOW}Critical failures: {len(critical_failures)}{Style.RESET_ALL}")
            for r in critical_failures:
                print(f"    - [{r['id']}] {r['prompt'][:60]}")
        avg_latency = sum(r.get("latency_ms", 0) for r in results) / total if total else 0
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"{'='*60}\n")

    def save_json(self, results: list, baseline: dict = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"eval_report_{timestamp}.json")
        report = {
            "timestamp": timestamp,
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "results": results,
        }
        if baseline:
            report["regression_vs_baseline"] = self._compare_baseline(results, baseline)

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {filename}")
        return filename

    def save_baseline(self, results: list) -> str:
        filename = os.path.join(self.output_dir, "baseline.json")
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "results": {r["id"]: {"passed": r["passed"], "score": r.get("score", 0)} for r in results},
        }
        with open(filename, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"  Baseline saved: {filename}")
        return filename

    def load_baseline(self) -> dict:
        filename = os.path.join(self.output_dir, "baseline.json")
        if not os.path.exists(filename):
            return None
        with open(filename) as f:
            return json.load(f)

    def _compare_baseline(self, results: list, baseline: dict) -> list:
        regressions = []
        baseline_results = baseline.get("results", {})
        for result in results:
            tc_id = result["id"]
            if tc_id in baseline_results:
                was_passing = baseline_results[tc_id]["passed"]
                now_passing = result["passed"]
                if was_passing and not now_passing:
                    regressions.append({
                        "id": tc_id,
                        "regression": "PASS -> FAIL",
                        "critical": result.get("critical", False),
                    })
                elif not was_passing and now_passing:
                    regressions.append({
                        "id": tc_id,
                        "regression": "FAIL -> PASS (improvement)",
                        "critical": False,
                    })
        return regressions
