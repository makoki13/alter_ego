"""
alterEgo - Generador de biografía.
Versión con RAG: busca registros relevantes antes de generar.
"""

from pathlib import Path
from typing import Any

from src.llm.base_provider import BaseLLMProvider
from src.rag.retriever import RAGRetriever


class BiographyGenerator:
    """Genera una biografía a partir de los datos procesados."""

    def __init__(self, llm: BaseLLMProvider, config: dict):
        self.llm = llm
        self.config = config
        self.output_path = Path(
            config.get("output", {}).get("biography_path", "output/biografia.md")
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, records: list[dict[str, Any]]) -> str:
        """Genera la biografía usando RAG."""

        rag_enabled = self.config.get("rag", {}).get("enabled", False)

        if rag_enabled:
            context_records = self._retrieve_with_rag(records)
        else:
            context_records = self._fallback_context(records)

        print(f"  📝 Registros para contexto: {len(context_records)}")

        # Construir contexto
        context = self._build_context(context_records)
        print(f"  📝 Contexto: {len(context)} caracteres")

        # Cargar system prompt
        system_prompt = self._load_system_prompt()

        # Preparar mensajes
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Aquí tienes la información REAL recolectada sobre esta persona. "
                "Úsala como base. NO inventes datos que no estén aquí.\n\n"
                f"{context}\n\n"
                "Escribe ahora su biografía completa basándote SOLO en estos datos."
            )},
        ]

        # Llamar al LLM
        print(f"  🤖 Generando biografía con {self.llm.get_model_name()}...")
        biography = self.llm.chat(messages, max_tokens=4096, temperature=0.7)

        # Guardar
        self._save(biography)
        return biography

    def _retrieve_with_rag(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Usa RAG para obtener los registros más relevantes."""
        retriever = RAGRetriever(self.config)

        # Intentar cargar índice existente
        if retriever.load_index():
            print("  📊 Índice RAG cargado desde disco.")
        else:
            print("  📊 No hay índice. Construyendo...")
            retriever.build_index(records)

        # Obtener queries temáticas de la configuración
        queries = self.config.get("rag", {}).get("biography_queries", [
            "infancia y familia",
            "trabajo y profesión",
            "aficiones y deportes",
        ])

        return retriever.retrieve_for_biography(queries)

    def _fallback_context(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fallback sin RAG: primeros N registros."""
        return records[:40]

    def _build_context(self, records: list[dict[str, Any]]) -> str:
        """Construye el texto de contexto a partir de registros."""
        parts = []

        for rec in records:
            source = rec.get("source", "?")
            rtype = rec.get("type", "?")
            title = rec.get("title", "")
            content = rec.get("content", "")[:400]
            date = rec.get("date", "")[:10]
            score = rec.get("_relevance_score", "")

            header = f"[{source}/{rtype}]"
            if date:
                header += f" [{date}]"
            if score:
                header += f" (relevancia: {score})"
            if title:
                header += f" {title}"

            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    def _load_system_prompt(self) -> str:
        prompt_path = Path("prompts/biography_system.md")
        if not prompt_path.exists():
            return (
                "Eres un biógrafo experto. Escribe en primera persona, en español. "
                "Sé fiel a los datos proporcionados. No inventes."
            )
        return prompt_path.read_text(encoding="utf-8")

    def _save(self, text: str):
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 Biografía guardada en: {self.output_path}")
