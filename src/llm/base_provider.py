"""
alterEgo - Interfaz base para proveedores LLM.
Permite cambiar de Groq a cualquier otro sin tocar el resto.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Interfaz común para todos los proveedores de LLM."""

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Envía mensajes al modelo y devuelve la respuesta como texto.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Devuelve el nombre del modelo activo."""
        pass
