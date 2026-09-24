"""
alterEgo - Factory de proveedores LLM.
Lee settings.yaml y devuelve el proveedor configurado.
"""

from src.llm.base_provider import BaseLLMProvider
from src.llm.groq_provider import GroqProvider


class LLMFactory:
    _providers = {
        "groq": GroqProvider,
        # Futuro: "ollama": OllamaProvider,
        # Futuro: "openai": OpenAIProvider,
    }

    @classmethod
    def get(cls, config: dict) -> BaseLLMProvider:
        provider_name = config.get("provider", "groq")
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            raise ValueError(
                f"Proveedor '{provider_name}' no soportado. "
                f"Disponibles: {list(cls._providers.keys())}"
            )

        return provider_class(config)
