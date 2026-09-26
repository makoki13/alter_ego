"""
alterEgo - Generador de embeddings.
Convierte texto en vectores numéricos usando sentence-transformers.
Modelo multilingüe optimizado para español.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer


class Embedder:
    """Genera embeddings de texto usando un modelo local."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"  🧠 Cargando modelo de embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  🧠 Modelo cargado. Dimensión: {self.dimension}")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Convierte una lista de textos en vectores.
        Devuelve lista de vectores (cada uno de dim 'dimension').
        """
        if not texts:
            return []

        print(f"  🧠 Generando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convierte una consulta en un vector."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        return self.dimension
