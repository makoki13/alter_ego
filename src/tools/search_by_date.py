"""
alterEgo - Herramienta: buscar por rango de fechas.
"""

from langchain_core.tools import tool

from src.memory.vector_memory import VectorMemory


def create_date_search_tool(memory: VectorMemory):
    """Crea herramienta de búsqueda temporal."""

    @tool
    def search_by_date(year: str, query: str = "") -> str:
        """Busca recuerdos de un año o época concreta.
        Úsala cuando pregunten sobre un periodo específico.
        Ejemplo: year='2019', query='vacaciones'"""
        # Búsqueda combinada: año + query
        search_query = f"{year} {query}".strip()
        results = memory.search(search_query, top_k=8)

        # Filtrar por año si es posible
        filtered = []
        for doc in results:
            doc_date = doc.metadata.get("date", "")
            if year in doc_date or not doc_date:
                filtered.append(doc)

        if not filtered:
            return f"No se encontraron recuerdos de {year}."

        parts = []
        for doc in filtered[:5]:
            parts.append(f"[{doc.metadata.get('date', '')[:10]}] {doc.page_content[:200]}")

        return "\n".join(parts)

    return search_by_date
