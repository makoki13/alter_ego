"""
alterEgo - Memoria vectorial usando FAISS + LangChain.
"""

from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorMemory:
    """Memoria a largo plazo de alterEgo basada en FAISS."""

    def __init__(self, config: dict):
        vs_config = config.get("vector_store", {})
        self.storage_path = Path(vs_config.get("path", "data/vectors"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        model_name = vs_config.get(
            "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore: FAISS | None = None

    def build_from_records(self, records: list[dict[str, Any]]):
        """Construye el índice FAISS desde registros procesados."""
        documents = []
        for rec in records:
            text = f"{rec.get('title', '')} {rec.get('content', '')}".strip()
            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "id": rec.get("id", ""),
                    "source": rec.get("source", ""),
                    "type": rec.get("type", ""),
                    "date": rec.get("date", ""),
                    "tags": rec.get("tags", []),
                },
            )
            documents.append(doc)

        if not documents:
            print("  ⚠ No hay documentos para indexar.")
            return

        print(f"  🧠 Indexando {len(documents)} documentos...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Guardar en disco
        self.vectorstore.save_local(str(self.storage_path))
        print(f"  💾 Índice guardado en: {self.storage_path}")

    def load(self) -> bool:
        """Carga un índice existente."""
        index_path = self.storage_path / "index.faiss"
        if not index_path.exists():
            return False

        self.vectorstore = FAISS.load_local(
            str(self.storage_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print("  📊 Índice FAISS cargado.")
        return True

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Búsqueda semántica."""
        if self.vectorstore is None:
            raise RuntimeError("No hay índice cargado. Ejecuta build o load primero.")
        return self.vectorstore.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 10):
        """Devuelve un retriever compatible con cadenas LangChain."""
        if self.vectorstore is None:
            raise RuntimeError("No hay índice cargado.")
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})
