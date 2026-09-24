"""
alterEgo - Generador de biografía.
Toma los registros procesados y genera una biografía usando el LLM.
"""

from pathlib import Path
from typing import Any

from src.llm.base_provider import BaseLLMProvider


class BiographyGenerator:
    """Genera una biografía a partir de los datos procesados."""

    def __init__(self, llm: BaseLLMProvider, config: dict):
        self.llm = llm
        self.output_path = Path(config.get("output", {}).get(
            "biography_path", "output/biografia.md"
        ))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, records: list[dict[str, Any]]) -> str:
        """Genera la biografía a partir de los registros."""

        print(f"  📚 Registros disponibles: {len(records)}")

        # Construir el contexto condensado
        context = self._build_context(records)
        print(f"  📝 Contexto construido: {len(context)} caracteres")

        # Cargar el system prompt
        system_prompt = self._load_system_prompt()

        # Preparar mensajes
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Aquí tienes toda la información recolectada sobre esta persona:\n\n"
                f"{context}\n\n"
                "Escribe ahora su biografía completa."
            )},
        ]

        # Llamar al LLM
        print(f"  🤖 Generando biografía con {self.llm.get_model_name()}...")
        biography = self.llm.chat(messages, max_tokens=4096, temperature=0.8)

        # Guardar
        self._save(biography)

        return biography

    def _build_context(self, records: list[dict[str, Any]]) -> str:
        """
        Condensa los registros en un texto que quepa en el contexto del LLM.
        Para el MVP: título + primeros 300 chars de cada registro.
        """
        parts = []

        # Separar por tipo para dar estructura
        documents = [r for r in records if r["type"] == "document"]
        images = [r for r in records if r["type"] == "image_metadata"]

        if documents:
            parts.append(f"=== DOCUMENTOS ({len(documents)}) ===")
            for rec in documents:
                title = rec.get("title", "Sin título")
                content = rec.get("content", "")[:300]
                date = rec.get("date", "")
                parts.append(f"[{date}] {title}: {content}")

        if images:
            parts.append(f"\n=== IMÁGENES ({len(images)}) ===")
            for rec in images:
                title = rec.get("title", "Sin título")
                content = rec.get("content", "")[:200]
                parts.append(f"- {title}: {content}")

        return "\n\n".join(parts)

    def _load_system_prompt(self) -> str:
        """Carga el prompt de sistema desde prompts/biography_system.md."""
        prompt_path = Path("prompts/biography_system.md")

        if not prompt_path.exists():
            # Prompt por defecto si no existe el archivo
            return (
                "Eres un biógrafo experto. Escribes biografías en primera persona, "
                "cálidas, detalladas y bien estructuradas. Usas la información "
                "proporcionada para construir una narrativa coherente. "
                "Si hay datos insuficientes, sé honesto y no inventes."
            )

        return prompt_path.read_text(encoding="utf-8")

    def _save(self, text: str):
        """Guarda la biografía en el archivo de salida."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 Biografía guardada en: {self.output_path}")
