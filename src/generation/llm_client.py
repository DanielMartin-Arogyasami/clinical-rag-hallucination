"""LLM client wrapper — model-agnostic via LiteLLM.

FIX M2: Added tenacity retry with exponential backoff for transient API failures.
"""
from __future__ import annotations

import logging
from typing import Any

from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from src.config import GenerationConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around LiteLLM for model-agnostic LLM calls."""

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages to LLM and return text response.

        FIX M2: Retries transient failures with exponential backoff.
        """
        import litellm

        model = self._resolve_model()
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            timeout=self.config.timeout_seconds,
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete_with_logprobs(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """Complete with logprobs for entropy-based confidence estimation."""
        import litellm

        model = self._resolve_model()
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            logprobs=True,
            top_logprobs=5,
            timeout=self.config.timeout_seconds,
        )
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "logprobs": getattr(choice, "logprobs", None),
        }

    def _resolve_model(self) -> str:
        provider = self.config.provider
        model = self.config.model
        if provider == "openai":
            return model
        elif provider == "anthropic":
            return f"anthropic/{model}" if not model.startswith("anthropic/") else model
        elif provider == "ollama":
            return f"ollama/{model}" if not model.startswith("ollama/") else model
        return model
