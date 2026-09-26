"""
alterEgo - Almacén vectorial con FAISS.
Guarda embeddings y permite búsqueda por similitud.
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    """Almacén de vectores basado en FAISS."""

    def __init__(self, storage_path: Path, dimension: int):
        self.storage_path = storage_path
        self.dimension = dimension
        self.index_path = storage_path / "index.faiss"
        self.map_path = storage_path / "records_map.json"
        self.meta_path = storage_path / "metadata.json"

        self.index: faiss.IndexFlatIP | None = None
        self.records_map: list[str] = []  # posición → record ID

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def build(self, embeddings: list[list[float]], record_ids: list[str]):
        """
        Construye el índice FAISS desde cero.
        embeddings: lista de vectores.
        record_ids: lista de IDs correspondientes (mismo orden).
        """
        if len(embeddings) != len(record_ids):
            raise ValueError("embeddings y record_ids deben tener la misma longitud")

        # Convertir a numpy
        vectors = np.array(embeddings, dtype=np.float32)

        # Crear índice de producto interno (cosine similarity
        # porque los vectores ya están normalizados)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.records_map = record_ids

        print(f"  📊 Índice FAISS construido: {self.index.ntotal} vectores")

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Busca los top_k vectores más similares.
        Devuelve lista de (record_id, score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record_id = self.records_map[idx]
            results.append((record_id, float(score)))

        return results

    def save(self):
        """Persiste el índice y el mapeo en disco."""
        if self.index is None:
            return

        faiss.write_index(self.index, str(self.index_path))

        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.records_map, f, ensure_ascii=False)

        metadata = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        print(f"  💾 Índice guardado en: {self.storage_path}")

    def load(self) -> bool:
        """Carga un índice existente. Devuelve True si se cargó correctamente."""
        if not self.index_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))

            with open(self.map_path, encoding="utf-8") as f:
                self.records_map = json.load(f)

            print(f"  📊 Índice cargado: {self.index.ntotal} vectores")
            return True

        except Exception as e:
            print(f"  ⚠ Error cargando índice: {e}")
            return False

    def exists(self) -> bool:
        """Comprueba si hay un índice guardado."""
        return self.index_path.exists() and self.map_path.exists()
