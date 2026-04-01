"""Constrained generation — enforces structured output via Instructor/Pydantic.

FIX M1: Uses system/user message separation with XML-delimited sections
        to mitigate prompt injection from source documents.
FIX M2: Inherits retry from Instructor's max_retries parameter.
"""
from __future__ import annotations

import logging
from typing import Any
from typing import TypeVar

from pydantic import BaseModel

from src.config import GenerationConfig
from src.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ConstrainedGenerator:
    """Generates schema-constrained extractions using Instructor + Pydantic."""

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        self._client: Any = None
        self._prompts: dict[str, str] = {}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import instructor

        if self.config.provider in ("openai", "azure"):
            from openai import OpenAI

            self._client = instructor.from_openai(OpenAI())
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic

            self._client = instructor.from_anthropic(Anthropic())
        else:
            import litellm

            self._client = instructor.from_litellm(litellm.completion)
        return self._client

    def load_prompt(self, name: str) -> str:
        if name not in self._prompts:
            settings = get_settings()
            path = settings.prompts_dir / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Prompt template not found: {path}")
            self._prompts[name] = path.read_text()
        return self._prompts[name]

    def extract(
        self,
        source_document: str,
        retrieved_context: str,
        response_model: type[T],
        task_description: str = "",
        prompt_name: str = "extraction",
    ) -> T:
        """Run constrained extraction.

        FIX M1: Uses system/user message split. System prompt contains rules;
        user prompt wraps source doc in XML tags to mitigate injection.
        """
        system_prompt = self.load_prompt(f"{prompt_name}_system")
        user_template = self.load_prompt(f"{prompt_name}_user")
        schema_desc = response_model.model_json_schema()

        user_prompt = user_template.format(
            source_document=source_document,
            retrieved_context=retrieved_context,
            target_schema=str(schema_desc),
            task_description=task_description,
        )

        client = self._get_client()
        logger.info(
            "Generating constrained extraction (model=%s, schema=%s)",
            self.config.model,
            response_model.__name__,
        )

        result = client.chat.completions.create(
            model=self.config.model,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
        )
        return result

    def extract_multiple(
        self,
        source_document: str,
        retrieved_context: str,
        response_model: type[T],
        task_description: str = "",
        n_samples: int = 1,
        temperature: float | None = None,
    ) -> list[T]:
        """Generate multiple extraction samples (for self-consistency).

        FIX H1: This method is now actually called from the pipeline orchestrator.
        """
        results: list[T] = []
        temp = temperature if temperature is not None else 0.7
        system_prompt = self.load_prompt("extraction_system")
        user_template = self.load_prompt("extraction_user")
        schema_desc = response_model.model_json_schema()
        user_prompt = user_template.format(
            source_document=source_document,
            retrieved_context=retrieved_context,
            target_schema=str(schema_desc),
            task_description=task_description,
        )
        client = self._get_client()

        for i in range(n_samples):
            try:
                result = client.chat.completions.create(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=self.config.max_tokens,
                    max_retries=2,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Self-consistency sample %d/%d failed: %s", i + 1, n_samples, e)
        return results
