import os
import json
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class NvidiaLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        
        self.model_nemotron_120b = os.getenv("NEMOTRON_120B_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        self.model_gpt_oss_120b = os.getenv("GPT_OSS_120B_MODEL", "openai/gpt-oss-120b")
        self.model_nemotron_550b = os.getenv("NEMOTRON_550B_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        self.default_model = os.getenv("DEFAULT_MODEL", self.model_nemotron_550b)

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY must be provided or set in environment variables.")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def call_nemotron_120b(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        reasoning_budget: int = 16384
    ) -> Generator[Dict[str, Any], None, None]:
        """Model 1: Nemotron 3 Super 120B with streaming reasoning."""
        completion = self.client.chat.completions.create(
            model=self.model_nemotron_120b,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": reasoning_budget},
            stream=True
        )

        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "content": reasoning}
            if delta.content is not None:
                yield {"type": "content", "content": delta.content}

    def call_gpt_oss_120b(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Model 2: OpenAI GPT OSS 120B non-streaming."""
        completion = self.client.chat.completions.create(
            model=self.model_gpt_oss_120b,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False
        )

        msg = completion.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        return {
            "reasoning": reasoning,
            "content": msg.content
        }

    def call_nemotron_550b(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        reasoning_budget: int = 16384
    ) -> Generator[Dict[str, Any], None, None]:
        """Model 3: Nemotron 3 Ultra 550B with streaming reasoning."""
        completion = self.client.chat.completions.create(
            model=self.model_nemotron_550b,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": reasoning_budget},
            stream=True
        )

        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "content": reasoning}
            if delta.content is not None:
                yield {"type": "content", "content": delta.content}


if __name__ == "__main__":
    print("Testing NVIDIA API Connection in Excercise Directory...")
    llm = NvidiaLLMClient()
    print("API Key loaded successfully.")
    print("Configured Models:")
    print(" 1. Nemotron 120B:", llm.model_nemotron_120b)
    print(" 2. GPT OSS 120B :", llm.model_gpt_oss_120b)
    print(" 3. Nemotron 550B:", llm.model_nemotron_550b)
