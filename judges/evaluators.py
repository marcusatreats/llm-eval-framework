import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


class KeywordJudge:
    """Evaluates responses based on keyword presence."""

    def evaluate(self, response: str, test_case: dict) -> dict:
        criteria = test_case.get("pass_criteria")
        response_lower = response.lower()

        if criteria == "must_contain_keyword":
            keywords = test_case.get("expected_keywords", [])
            matched = [kw for kw in keywords if kw.lower() in response_lower]
            passed = len(matched) == len(keywords) and len(keywords) > 0
            return {
                "judge": "keyword",
                "passed": passed,
                "score": len(matched) / len(keywords) if keywords else 0.0,
                "detail": f"Matched {len(matched)}/{len(keywords)} keywords: {matched}",
            }

        elif criteria == "keyword_threshold":
            keywords = test_case.get("expected_keywords", [])
            threshold = test_case.get("keyword_threshold", 1)
            matched = [kw for kw in keywords if kw.lower() in response_lower]
            passed = len(matched) >= threshold
            return {
                "judge": "keyword",
                "passed": passed,
                "score": len(matched) / len(keywords) if keywords else 0.0,
                "detail": f"Matched {len(matched)}/{len(keywords)}, threshold={threshold}: {matched}",
            }

        elif criteria == "must_not_contain_keyword":
            forbidden = test_case.get("forbidden_keywords", [])
            found = [kw for kw in forbidden if kw.lower() in response_lower]
            passed = len(found) == 0
            return {
                "judge": "keyword",
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "detail": f"Forbidden keywords found: {found}" if found else "No forbidden keywords found",
            }

        elif criteria == "non_empty_response":
            passed = len(response.strip()) > 0
            return {
                "judge": "keyword",
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "detail": "Response is non-empty" if passed else "Response was empty",
            }

        return {
            "judge": "keyword",
            "passed": False,
            "score": 0.0,
            "detail": f"Unknown criteria: {criteria}",
        }


class LLMJudge:
    """Uses another LLM call to evaluate response quality."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
        self.threshold = float(os.getenv("LLM_JUDGE_THRESHOLD", "0.7"))

    def evaluate(self, prompt: str, response: str, test_case: dict) -> dict:
        rubric = test_case.get("judge_rubric", "Evaluate if the response is accurate and helpful.")
        judge_prompt = f"""You are an evaluator assessing the quality of an AI response.

Original prompt: {prompt}
AI Response: {response}

Rubric: {rubric}

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reasoning": "your reasoning here"}}

Score must be between 0.0 and 1.0."""

        try:
            result = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            import json
            text = result.content[0].text.strip()
            # Strip markdown if present
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            passed = score >= self.threshold
            return {
                "judge": "llm",
                "passed": passed,
                "score": score,
                "detail": parsed.get("reasoning", ""),
            }
        except Exception as e:
            return {
                "judge": "llm",
                "passed": False,
                "score": 0.0,
                "detail": f"Judge error: {str(e)}",
            }


class LatencyJudge:
    """Checks if response latency is within acceptable bounds."""

    def evaluate(self, latency_ms: int, test_case: dict) -> dict:
        max_latency = test_case.get("max_latency_ms")
        if max_latency is None:
            return {"judge": "latency", "passed": True, "score": 1.0, "detail": "No latency requirement"}

        passed = latency_ms <= max_latency
        return {
            "judge": "latency",
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "detail": f"{latency_ms}ms vs {max_latency}ms limit",
        }
