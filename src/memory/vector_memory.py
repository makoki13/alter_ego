"""
alterEgo - Memoria vectorial usando FAISS + LangChain.
Almacena embeddings de todos los registros y permite búsqueda semántica.
"""

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorMemory:
    """Memoria a largo plazo de alterEgo basada en FAISS."""

    def __init__(self, config: dict):
        vs_config = config.get("vector_store", {})
        self.storage_path = Path(vs_config.get("path", "data/vectors"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        model_name = vs_config.get(
            "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.batch_size = vs_config.get("batch_size", 64)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore: FAISS | None = None

    def build_from_records(self, records: list[dict[str, Any]]) -> None:
        """
        Convierte registros procesados en documentos LangChain
        y construye el índice FAISS.
        """
        documents: list[Document] = []

        for rec in records:
            title = rec.get("title", "")
            content = rec.get("content", "")
            text = f"{title} {content}".strip()

            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "id": rec.get("id", ""),
                    "source": rec.get("source", ""),
                    "type": rec.get("type", ""),
                    "date": rec.get("date", ""),
                    "tags": ", ".join(rec.get("tags", [])),
                },
            )
            documents.append(doc)

        if not documents:
            print("  ⚠ No hay documentos para indexar.")
            return

        print(f"  🧠 Indexando {len(documents)} documentos...")
        print(f"     Modelo: {self.embeddings.model_name}")
        print(f"     Batch size: {self.batch_size}")

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings,
        )

        self.vectorstore.save_local(str(self.storage_path))
        print(f"  💾 Índice guardado en: {self.storage_path}")

    def load(self) -> bool:
        """Carga un índice FAISS existente desde disco."""
        index_path = self.storage_path / "index.faiss"
        if not index_path.exists():
            return False

        try:
            self.vectorstore = FAISS.load_local(
                str(self.storage_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"  📊 Índice FAISS cargado desde: {self.storage_path}")
            return True
        except Exception as e:
            print(f"  ❌ Error cargando índice: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Búsqueda semántica: devuelve los top_k documentos más similares."""
        if self.vectorstore is None:
            raise RuntimeError(
                "No hay índice cargado. Ejecuta: python main.py --build-index"
            )
        return self.vectorstore.similarity_search(query, k=top_k)

    def search_with_scores(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """Búsqueda con puntuación de similitud."""
        if self.vectorstore is None:
            raise RuntimeError(
                "No hay índice cargado. Ejecuta: python main.py --build-index"
            )
        return self.vectorstore.similarity_search_with_score(query, k=top_k)

    def as_retriever(self, top_k: int = 10):
        """
        Devuelve un retriever compatible con cadenas LangChain.
        Útil para RetrievalQA, create_retrieval_chain, etc.
        """
        if self.vectorstore is None:
            raise RuntimeError("No hay índice cargado.")
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    def get_document_count(self) -> int:
        """Devuelve el número de documentos en el índice."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal
