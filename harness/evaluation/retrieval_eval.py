"""
alterEgo - Evaluación de la calidad del retrieval (FAISS).
Mide: precision@k, recall, relevancia de resultados.
"""

import json
from pathlib import Path
from typing import Any
from src.memory.vector_memory import VectorMemory


def evaluate_retrieval(config: dict, golden_path: Path) -> dict[str, Any]:
    """Evalúa el retrieval contra consultas doradas."""

    # Cargar consultas doradas
    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)
    queries = golden_data["queries"]

    # Cargar memoria
    memory = VectorMemory(config)
    if not memory.load():
        print("  ❌ No hay índice. Ejecuta: python main.py --build-index")
        return {"error": "no_index"}

    results = []
    total_precision = 0.0
    total_queries_with_expectations = 0

    print(f"\n  📊 Evaluando retrieval con {len(queries)} consultas...\n")

    for item in queries:
        query = item["query"]
        expected_keywords = [kw.lower() for kw in item.get("expected_keywords", [])]
        min_results = item.get("min_relevant_results", 0)

        # Ejecutar búsqueda
        docs_with_scores = memory.search_with_scores(query, top_k=10)

        # Evaluar
        hits = 0
        retrieved_texts = []
        for doc, score in docs_with_scores:
            content = doc.page_content.lower()
            retrieved_texts.append(content)
            if any(kw in content for kw in expected_keywords):
                hits += 1

        total_retrieved = len(docs_with_scores)
        precision = hits / total_retrieved if total_retrieved > 0 else 0.0

        # Para queries que esperan resultados
        if min_results > 0:
            passed = hits >= min_results
            total_precision += precision
            total_queries_with_expectations += 1
        else:
            # Queries que NO esperan resultados: pasamos si devuelve pocos o ningún hit
            passed = hits <= 1  # Permitimos 1 hit accidental
            total_queries_with_expectations += 1
            total_precision += (1.0 - precision)  # Invertimos: menos hits = mejor

        status = "✅" if passed else "❌"
        print(f"  {status} [{item['id']}] '{query[:40]}...'")
        print(f"      → {total_retrieved} resultados, {hits} relevantes (esperados ≥{min_results})")

        results.append({
            "id": item["id"],
            "query": query,
            "retrieved_count": total_retrieved,
            "relevant_hits": hits,
            "precision": round(precision, 3),
            "passed": passed,
            "difficulty": item.get("difficulty", "medium"),
        })

    # Métricas globales
    avg_precision = total_precision / total_queries_with_expectations if total_queries_with_expectations > 0 else 0
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    summary = {
        "total_queries": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "avg_precision": round(avg_precision, 3),
        "pass_rate": round(passed_count / len(results), 3) if results else 0,
        "details": results,
    }

    print(f"\n  {'─' * 50}")
    print(f"  📊 Precisión media: {avg_precision:.3f}")
    print(f"  📊 Pasaron: {passed_count}/{len(results)} ({summary['pass_rate']*100:.0f}%)")
    print(f"  {'─' * 50}\n")

    return summary