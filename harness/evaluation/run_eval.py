"""
alterEgo - Ejecutor del arnés de evaluación.
Uso:
  python harness/evaluation/run_eval.py --retrieval
  python harness/evaluation/run_eval.py --generation
  python harness/evaluation/run_eval.py --all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime

import yaml


def load_config() -> dict:
    with open("config/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="alterEgo - Evaluación")
    parser.add_argument("--retrieval", action="store_true", help="Evaluar retrieval FAISS")
    parser.add_argument("--generation", action="store_true", help="Evaluar generación (LLM-as-judge)")
    parser.add_argument("--all", action="store_true", help="Evaluar todo")
    args = parser.parse_args()

    config = load_config()
    golden_path = Path("harness/evaluation/golden_queries.json")

    if not golden_path.exists():
        print("❌ No se encuentra golden_queries.json")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    print(f"\n{'=' * 60}")
    print("  🎭 alterEgo - Arnés de evaluación")
    print(f"  Run ID: {run_id}")
    print(f"{'=' * 60}")

    # Evaluar retrieval
    if args.retrieval or args.all:
        from harness.evaluation.retrieval_eval import evaluate_retrieval
        results["retrieval"] = evaluate_retrieval(config, golden_path)

    # Evaluar generación
    if args.generation or args.all:
        from harness.evaluation.generation_eval import run_generation_eval
        results["generation"] = run_generation_eval(config, golden_path)

    if not results:
        parser.print_help()
        return

    # Guardar resultados
    reports_dir = Path("harness/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"eval_{run_id}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"  💾 Informe guardado en: {report_path}")

    # Resumen final
    print(f"\n{'=' * 60}")
    print("  RESUMEN FINAL")
    print(f"{'=' * 60}")

    if "retrieval" in results:
        r = results["retrieval"]
        print(f"  Retrieval: {r.get('passed', '?')}/{r.get('total_queries', '?')} pasaron "
              f"(precisión: {r.get('avg_precision', '?')})")

    if "generation" in results:
        g = results["generation"]
        print(f"  Generación: puntuación media {g.get('avg_overall_score', '?')}/5")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()