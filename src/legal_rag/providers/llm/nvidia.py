"""
LLM Provider abstraction + NVIDIA implementation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generator

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...

    @abstractmethod
    def generate_structured(
        self, messages: list[dict[str, str]], response_schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]: ...

    @abstractmethod
    def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]: ...


class NvidiaLLMProvider(LLMProvider):
    """
    NVIDIA LLM provider using the OpenAI-compatible API.
    Supports reasoning models (Nemotron 120B, 550B) and standard models (GPT OSS 120B).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            max_retries=max_retries,
        )
        self._is_reasoning_model = any(
            kw in model for kw in ["nemotron-3-super", "nemotron-3-ultra", "nemotron-ultra", "nemotron-super"]
        )
        logger.info("NvidiaLLMProvider: model=%s reasoning=%s", model, self._is_reasoning_model)

    @property
    def model(self) -> str:
        return self._model

    def _call(
        self, messages: list[dict[str, str]], stream: bool = False, max_tokens: int | None = None
    ):
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            stream=stream,
        )
        if self._is_reasoning_model:
            budget = kwargs.get("reasoning_budget") or 1024
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": min(budget, 1024),
            }
        return self._client.chat.completions.create(**kwargs)


    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Non-streaming generation. Returns only the answer content (not reasoning trace)."""
        resp = self._call(messages, stream=False, max_tokens=kwargs.get("max_tokens"))
        msg = resp.choices[0].message
        content = msg.content or ""
        return content.strip()

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a structured response.
        Instructs the model to respond in JSON matching response_schema.
        """
        import json
        schema_str = json.dumps(response_schema, indent=2)
        system_injection = (
            f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_str}"
        )
        augmented = list(messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = augmented[0].copy()
            augmented[0]["content"] += system_injection
        else:
            augmented.insert(0, {"role": "system", "content": system_injection.strip()})

        raw = self.generate(augmented, **kwargs)
        # Extract JSON from potential markdown fences
        import re
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_response": raw}

    def stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        """Stream content tokens only (not reasoning traces)."""
        stream_resp = self._call(messages, stream=True, max_tokens=kwargs.get("max_tokens"))
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


def get_llm_provider_by_model(config, model_name: str) -> LLMProvider:
    """Factory: create provider for specified model_name."""
    if config.rag_llm_provider == "nvidia":
        return NvidiaLLMProvider(
            model=model_name,
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url,
            max_tokens=config.rag_llm_max_tokens,
            temperature=config.rag_llm_temperature,
            timeout=config.rag_llm_timeout,
            max_retries=config.rag_llm_max_retries,
        )
    raise ValueError(f"Unknown LLM provider: {config.rag_llm_provider}")


def get_llm_provider(config) -> LLMProvider:
    """Factory: create default provider from RagConfig."""
    return get_llm_provider_by_model(config, config.rag_llm_model)

