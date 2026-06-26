# LLM Eval Framework

A Python-based evaluation harness for testing Large Language Model outputs — covering functional correctness, prompt regression detection, consistency testing, adversarial inputs, and latency validation.

## Features

- **Golden dataset** — curated test cases with expected outputs and pass criteria
- **Multiple judges** — keyword matching, LLM-as-judge scoring, latency checks
- **Consistency testing** — runs same prompt N times to detect non-determinism
- **Regression detection** — compare against saved baselines to catch drift
- **Pytest integration** — run as standard test suite in CI/CD
- **Adversarial testing** — prompt injection and boundary case coverage

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

### Run full eval suite
```bash
python eval_runner.py
```

### Run specific category
```bash
python eval_runner.py --category factual
python eval_runner.py --category adversarial
python eval_runner.py --critical-only
```

### Save baseline and compare
```bash
# Save current results as baseline
python eval_runner.py --save-baseline

# Compare future runs against baseline (regression detection)
python eval_runner.py --compare-baseline
```

### Run via pytest (CI/CD integration)
```bash
pytest tests/test_llm_eval.py -v
pytest tests/test_llm_eval.py -v -k "critical"
```

## Project Structure

```
llm-eval-framework/
├── eval_runner.py          # Main runner with CLI
├── requirements.txt
├── .env.example
├── data/
│   └── golden_dataset.json # Test cases
├── judges/
│   └── evaluators.py       # Keyword, LLM, and latency judges
├── utils/
│   └── llm_client.py       # Anthropic API wrapper
├── reports/
│   ├── report_generator.py # Console + JSON reporting
│   └── baseline.json       # Saved baseline (generated)
└── tests/
    └── test_llm_eval.py    # Pytest suite
```

## Adding Test Cases

Edit `data/golden_dataset.json`. Each test case supports:

```json
{
  "id": "TC009",
  "category": "factual|reasoning|boundary|consistency|llm_judge|adversarial|performance",
  "prompt": "Your prompt here",
  "expected_keywords": ["keyword1", "keyword2"],
  "forbidden_keywords": ["bad_word"],
  "pass_criteria": "must_contain_keyword|keyword_threshold|must_not_contain_keyword|non_empty_response|llm_judge",
  "keyword_threshold": 2,
  "runs": 1,
  "max_latency_ms": 5000,
  "judge_rubric": "For llm_judge: describe scoring criteria",
  "critical": false
}
```

## Pass Criteria

| Criteria | Description |
|---|---|
| `must_contain_keyword` | All expected_keywords must appear in response |
| `keyword_threshold` | At least N keywords must appear |
| `must_not_contain_keyword` | No forbidden_keywords may appear |
| `non_empty_response` | Response must not be empty |
| `llm_judge` | Another LLM scores the response using judge_rubric |

## CI/CD Integration

Exit code is `1` if any critical test cases fail — compatible with GitHub Actions, Jenkins, etc.

```yaml
# GitHub Actions example
- name: Run LLM Evals
  run: python eval_runner.py --compare-baseline
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```
