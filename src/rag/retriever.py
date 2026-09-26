"""
alterEgo - Retriever RAG.
Combina embedder + vector store para buscar registros relevantes.
"""

import json
from pathlib import Path
from typing import Any

from src.rag.embedder import Embedder
from src.rag.vector_store import FAISSVectorStore


class RAGRetriever:
    """Sistema de recuperación aumentada para alterEgo."""

    def __init__(self, config: dict):
        vs_config = config.get("vector_store", {})
        rag_config = config.get("rag", {})

        self.embedding_model = vs_config.get(
            "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.dimension = vs_config.get("dimension", 384)
        self.batch_size = vs_config.get("batch_size", 64)
        self.storage_path = Path(vs_config.get("path", "data/vectors"))
        self.top_k = rag_config.get("top_k", 15)

        self.embedder: Embedder | None = None
        self.store = FAISSVectorStore(self.storage_path, self.dimension)
        self.records: dict[str, dict[str, Any]] = {}  # id → record

    def build_index(self, records: list[dict[str, Any]]):
        """
        Genera embeddings de todos los registros y construye el índice.
        """
        if not records:
            print("  ⚠ No hay registros para indexar.")
            return

        # Guardar referencia a los registros
        self.records = {r["id"]: r for r in records}

        # Preparar textos para embedding
        texts = []
        record_ids = []
        for rec in records:
            # Combinar título + contenido para un embedding más rico
            text = f"{rec.get('title', '')} {rec.get('content', '')}".strip()
            if text:
                texts.append(text)
                record_ids.append(rec["id"])

        print(f"\n  📚 Indexando {len(texts)} registros...")

        # Inicializar embedder
        self.embedder = Embedder(self.embedding_model)

        # Generar embeddings
        embeddings = self.embedder.embed_texts(texts, self.batch_size)

        # Construir índice FAISS
        self.store.build(embeddings, record_ids)

        # Guardar en disco
        self.store.save()

        # Guardar también los registros completos para acceso rápido
        records_path = self.storage_path / "all_records.json"
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print("  ✅ Índice construido y guardado.")

    def load_index(self) -> bool:
        """Carga un índice existente."""
        if not self.store.load():
            return False

        # Cargar registros
        records_path = self.storage_path / "all_records.json"
        if records_path.exists():
            with open(records_path, encoding="utf-8") as f:
                records_list = json.load(f)
            self.records = {r["id"]: r for r in records_list}
            return True

        return False

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Busca los registros más relevantes para una consulta.
        Devuelve los registros completos ordenados por relevancia.
        """
        if self.embedder is None:
            self.embedder = Embedder(self.embedding_model)

        if top_k is None:
            top_k = self.top_k

        # Embedding de la consulta
        query_vector = self.embedder.embed_query(query)

        # Búsqueda en FAISS
        results = self.store.search(query_vector, top_k)

        # Devolver registros completos
        retrieved = []
        for record_id, score in results:
            if record_id in self.records:
                rec = self.records[record_id].copy()
                rec["_relevance_score"] = round(score, 4)
                retrieved.append(rec)

        return retrieved

    def retrieve_for_biography(
        self, queries: list[str], top_k_per_query: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Realiza múltiples búsquedas temáticas y combina resultados.
        Elimina duplicados y ordena por relevancia.
        """
        if top_k_per_query is None:
            top_k_per_query = self.top_k

        all_records: dict[str, dict[str, Any]] = {}

        for query in queries:
            print(f"  🔍 Buscando: '{query[:50]}...'")
            results = self.retrieve(query, top_k_per_query)
            for rec in results:
                rid = rec["id"]
                # Si ya existe, quedamos con el de mayor score
                if rid not in all_records or rec.get("_relevance_score", 0) > all_records[rid].get("_relevance_score", 0):
                    all_records[rid] = rec

        # Ordenar por relevancia descendente
        sorted_records = sorted(
            all_records.values(),
            key=lambda r: r.get("_relevance_score", 0),
            reverse=True,
        )

        print(f"  📚 Total de registros únicos recuperados: {len(sorted_records)}")
        return sorted_records
