import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Wrapper around Anthropic API with latency tracking."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = os.getenv("EVAL_MODEL", "claude-haiku-4-5-20251001")

    def call(self, prompt: str, system_prompt: str = None, max_tokens: int = 500) -> dict:
        """
        Send a prompt to the LLM and return response with metadata.

        Returns:
            dict with keys: response, latency_ms, model, tokens_used
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.time()
        try:
            result = self.client.messages.create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)
            return {
                "response": result.content[0].text,
                "latency_ms": latency_ms,
                "model": result.model,
                "tokens_used": result.usage.input_tokens + result.usage.output_tokens,
                "error": None,
            }
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return {
                "response": "",
                "latency_ms": latency_ms,
                "model": self.model,
                "tokens_used": 0,
                "error": str(e),
            }
