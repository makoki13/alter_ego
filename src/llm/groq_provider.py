"""
alterEgo - Proveedor Groq.
Implementa BaseLLMProvider usando la API de GroqCloud.
"""

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from src.llm.base_provider import BaseLLMProvider

load_dotenv()


class GroqProvider(BaseLLMProvider):
    """Proveedor LLM que usa GroqCloud."""

    def __init__(self, config: dict):
        api_key = os.getenv(config.get("api_key_env", "GROQ_API_KEY"))
        if not api_key:
            raise ValueError(
                "No se encontró la API key. "
                "Revisa tu archivo .env y settings.yaml."
            )
        self.client = Groq(api_key=api_key)
        self.model = config.get("model", "llama-3.3-70b-versatile")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.7)

    def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return f"Groq/{self.model}"
