"""
alterEgo - Herramienta: buscar en la memoria.
"""

from langchain_core.tools import tool

from src.memory.vector_memory import VectorMemory


def create_search_tool(memory: VectorMemory):
    """Crea la herramienta de búsqueda con acceso a la memoria."""

    @tool
    def search_memory(query: str) -> str:
        """Busca en los recuerdos y datos personales del usuario.
        Úsala cuando necesites información sobre la vida, experiencias,
        personas, lugares o eventos del usuario."""
        results = memory.search(query, top_k=5)
        if not results:
            return "No se encontraron recuerdos relevantes."

        parts = []
        for doc in results:
            source = doc.metadata.get("source", "?")
            date = doc.metadata.get("date", "")[:10]
            parts.append(f"[{source}] [{date}] {doc.page_content[:300]}")

        return "\n\n".join(parts)

    return search_memory
