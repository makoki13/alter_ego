"""
alterEgo - Evaluación de la calidad de generación (LLM-as-judge).
Usa el propio LLM para evaluar si las respuestas son fieles.
"""

import json
from pathlib import Path
from typing import Any

from src.llm.provider import get_llm
from src.memory.vector_memory import VectorMemory


JUDGE_PROMPT = """Eres un evaluador de calidad. Tu tarea es evaluar si una respuesta
de un asistente virtual es FIEL a los datos proporcionados.

## Criterios de evaluación (puntúa cada uno de 1 a 5):
1. **Fidelidad**: ¿La respuesta se basa SOLO en los datos proporcionados?
2. **No invención**: ¿La respuesta inventa datos, nombres, fechas o eventos?
3. **Honestidad**: Si no hay datos, ¿el asistente lo reconoce?
4. **Relevancia**: ¿La respuesta responde a la pregunta?
5. **Tono**: ¿Suena natural y cercano?

## Datos proporcionados al asistente:
{context}

## Pregunta del usuario:
{question}

## Respuesta del asistente:
{answer}

## Tu evaluación:
Responde SOLO en formato JSON:
{{
  "fidelidad": <1-5>,
  "no_invencion": <1-5>,
  "honestidad": <1-5>,
  "relevancia": <1-5>,
  "tono": <1-5>,
  "overall": <1-5>,
  "comentario": "<breve justificación>",
  "alucinaciones_detectadas": ["<dato inventado 1>", "<dato inventado 2>"]
}}"""


def evaluate_generation(
    config: dict,
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """Evalúa una respuesta del agente usando LLM-as-judge."""

    llm = get_llm(config)

    prompt = JUDGE_PROMPT.format(
        context=context[:2000],
        question=question,
        answer=answer[:2000],
    )

    response = llm.invoke(prompt)
    response_text = response.content

    # Intentar parsear como JSON
    try:
        # Buscar el JSON en la respuesta
        start = response_text.find("{") # type: ignore
        end = response_text.rfind("}") + 1 # type: ignore
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str) # type: ignore
    except (json.JSONDecodeError, ValueError):
        pass

    # Si no se puede parsear, devolver la respuesta raw
    return {"raw_response": response_text, "error": "No se pudo parsear JSON"}


def run_generation_eval(config: dict, golden_path: Path) -> dict[str, Any]:
    """Evalúa la generación para un subconjunto de queries doradas."""
    from src.agent.alter_ego_agent import AlterEgoAgent

    # Cargar queries doradas
    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)

    # Solo evaluar queries de dificultad "easy" y "medium"
    queries = [q for q in golden_data["queries"] if q.get("difficulty") != "hard"]

    if not queries:
        print("  ⚠ No hay queries para evaluar generación.")
        return {}

    # Limitar a 3 queries para no gastar muchos tokens
    queries = queries[:3]

    print(f"\n  📊 Evaluando generación con {len(queries)} consultas...\n")

    # Inicializar agente
    agent = AlterEgoAgent(config)
    agent.setup()

    # Inicializar memoria para obtener contexto
    memory = VectorMemory(config)
    memory.load()

    results = []
    for item in queries:
        question = item["query"]
        print(f"  🔄 Evaluando: '{question[:40]}...'")

        # Obtener respuesta del agente
        answer = agent.chat(question)

        # Obtener contexto (lo que FAISS devuelve para esa query)
        docs = memory.search(question, top_k=3)
        context = "\n\n".join(doc.page_content[:200] for doc in docs)

        # Evaluar con LLM-as-judge
        eval_result = evaluate_generation(config, question, answer, context)

        results.append({
            "id": item["id"],
            "question": question,
            "answer_preview": answer[:200],
            "evaluation": eval_result,
        })

        overall = eval_result.get("overall", "?")
        comment = eval_result.get("comentario", "")[:80]
        print(f"      → Overall: {overall}/5 | {comment}")

    # Resumen
    overalls = [r["evaluation"].get("overall", 0) for r in results if isinstance(r["evaluation"].get("overall"), (int, float))]
    avg_overall = sum(overalls) / len(overalls) if overalls else 0

    summary = {
        "total_evaluated": len(results),
        "avg_overall_score": round(avg_overall, 2),
        "details": results,
    }

    print(f"\n  {'─' * 50}")
    print(f"  📊 Puntuación media: {avg_overall:.2f}/5")
    print(f"  {'─' * 50}\n")

    return summary