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
        Ejemplo: year='2019', query='vacaciones'
        Ejemplo: year='2022', query='trabajo'"""
        search_query = f"{year} {query}".strip()
        results = memory.search(search_query, top_k=8)

        # Filtrar por año si es posible
        filtered = []
        for doc in results:
            doc_date = doc.metadata.get("date", "")
            if year in doc_date or not doc_date:
                filtered.append(doc)

        # Si el filtro deja muy pocos, usar todos
        if len(filtered) < 2:
            filtered = results

        if not filtered:
            return f"No se encontraron recuerdos de {year}."

        parts = []
        for doc in filtered[:5]:
            date = doc.metadata.get("date", "")[:10]
            source = doc.metadata.get("source", "?")
            parts.append(f"[{source}] [{date}] {doc.page_content[:250]}")

        return "\n\n".join(parts)

    return search_by_date
