"""
alterEgo - Proveedor LLM usando LangChain + Groq.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm(config: dict) -> ChatGroq:
    """Devuelve una instancia de ChatGroq configurada."""
    llm_config = config.get("llm", {})

    return ChatGroq(
        model=llm_config.get("model", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
    )
