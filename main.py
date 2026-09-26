"""
alterEgo - Punto de entrada principal.
Uso:
  python main.py --collect disk
  python main.py --collect telegram
  python main.py --collect x_twitter
  python main.py --collect all
  python main.py --build-index
  python main.py --generate biography
  python main.py --query "¿Dónde veraneaba de pequeño?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import yaml


def load_config() -> dict:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        print("❌ No se encuentra config/settings.yaml")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_records() -> list[dict]:
    """Carga y combina todos los registros procesados."""
    processed_dir = Path("data/processed")
    all_records = []

    for json_file in sorted(processed_dir.glob("*_records.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                records = json.load(f)
            print(f"  📂 {json_file.name}: {len(records)} registros")
            all_records.extend(records)
        except Exception as e:
            print(f"  ⚠ Error leyendo {json_file.name}: {e}")

    print(f"  📚 Total combinado: {len(all_records)} registros\n")
    return all_records


# ── COLECTORES ──

def run_disk_collector(config: dict):
    from src.collectors.disk_collector import DiskCollector

    source_cfg = config["sources"]["disk"]
    if not source_cfg.get("enabled", False):
        print("❌ Colector de disco desactivado.")
        return

    collector = DiskCollector(
        source_cfg, Path("data/raw/disk"), Path("data/processed")
    )

    print(f"\n{'=' * 50}")
    print(f"  alterEgo - Colector: {collector.name}")
    print(f"{'=' * 50}")

    print("\n[1/2] Recolectando archivos...")
    count = collector.collect()
    if count == 0:
        print("⚠ No se encontraron archivos.")
        return

    print("\n[2/2] Procesando archivos...")
    records = collector.process()
    if not records:
        return

    output_file = Path("data/processed/disk_records.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado en: {output_file}  ({len(records)} registros)")


def run_x_collector(config: dict):
    from src.collectors.x_collector import XCollector

    source_cfg = config["sources"]["x_twitter"]
    if not source_cfg.get("enabled", False):
        print("❌ Colector de X desactivado.")
        return

    collector = XCollector(
        source_cfg, Path("data/raw/x_twitter"), Path("data/processed")
    )

    print(f"\n{'=' * 50}")
    print(f"  alterEgo - Colector: {collector.name}")
    print(f"{'=' * 50}")

    print("\n[1/2] Verificando export...")
    count = collector.collect()
    if count == 0:
        return

    print("\n[2/2] Procesando datos de X...")
    records = collector.process()
    if not records:
        return

    output_file = Path("data/processed/x_records.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado en: {output_file}  ({len(records)} registros)")


def run_telegram_collector(config: dict):
    from src.collectors.telegram_collector import TelegramCollector

    source_cfg = config["sources"]["telegram"]
    if not source_cfg.get("enabled", False):
        print("❌ Colector de Telegram desactivado.")
        return

    collector = TelegramCollector(
        source_cfg, Path("data/raw/telegram"), Path("data/processed")
    )

    print(f"\n{'=' * 50}")
    print(f"  alterEgo - Colector: {collector.name}")
    print(f"{'=' * 50}")

    print("\n[1/2] Verificando export...")
    count = collector.collect()
    if count == 0:
        return

    print("\n[2/2] Procesando mensajes...")
    records = collector.process()
    if not records:
        return

    output_file = Path("data/processed/telegram_records.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado en: {output_file}  ({len(records)} registros)")


# ── RAG ──

def run_build_index(config: dict):
    """Construye el índice vectorial con todos los registros."""
    from src.rag.retriever import RAGRetriever

    print(f"\n{'=' * 50}")
    print("  alterEgo - Construir índice RAG")
    print(f"{'=' * 50}\n")

    records = load_all_records()
    if not records:
        print("❌ No hay registros. Ejecuta primero: python main.py --collect all")
        return

    retriever = RAGRetriever(config)
    retriever.build_index(records)

    print("\n✅ Índice RAG construido correctamente.")
    print("   Ya puedes usar: python main.py --generate biography")


def run_query(config: dict, query: str):
    """Prueba una consulta contra el índice RAG."""
    from src.rag.retriever import RAGRetriever

    print(f"\n{'=' * 50}")
    print("  alterEgo - Consulta RAG")
    print(f"{'=' * 50}\n")

    retriever = RAGRetriever(config)
    if not retriever.load_index():
        print("❌ No hay índice. Ejecuta primero: python main.py --build-index")
        return

    results = retriever.retrieve(query, top_k=5)

    print(f"\n  Consulta: \"{query}\"")
    print(f"  Resultados: {len(results)}\n")

    for i, rec in enumerate(results, 1):
        score = rec.get("_relevance_score", "?")
        source = rec.get("source", "?")
        content = rec.get("content", "")[:150]
        print(f"  [{i}] ({score}) [{source}] {content}...")
        print()


# ── GENERADORES ──

def run_biography_generator(config: dict):
    from src.generators.biography import BiographyGenerator
    from src.llm.factory import LLMFactory

    print(f"\n{'=' * 50}")
    print("  alterEgo - Generador de biografía")
    print(f"{'=' * 50}\n")

    records = load_all_records()
    if not records:
        print("❌ No hay registros. Ejecuta primero: python main.py --collect all")
        return

    llm_config = config.get("llm", {})
    llm = LLMFactory.get(llm_config)
    print(f"  🔌 Proveedor: {llm.get_model_name()}")

    generator = BiographyGenerator(llm, config)
    biography = generator.generate(records)

    print(f"\n{'=' * 50}")
    print("  Vista previa (primeros 500 chars):")
    print(f"{'=' * 50}")
    print(biography[:500])
    print("...")


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser(description="alterEgo - Tu gemelo digital")
    parser.add_argument(
        "--collect",
        type=str,
        choices=["disk", "x_twitter", "telegram", "gdrive", "all"],
        help="Fuente a recolectar",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Construir índice RAG (FAISS)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Consultar el índice RAG",
    )
    parser.add_argument(
        "--generate",
        type=str,
        choices=["biography"],
        help="Qué generar",
    )
    args = parser.parse_args()

    config = load_config()
    print(f"\n🎭 alterEgo v{config['project']['version']}")

    if args.collect == "disk":
        run_disk_collector(config)
    elif args.collect == "x_twitter":
        run_x_collector(config)
    elif args.collect == "telegram":
        run_telegram_collector(config)
    elif args.collect == "all":
        run_disk_collector(config)
        run_x_collector(config)
        run_telegram_collector(config)
    elif args.build_index:
        run_build_index(config)
    elif args.query:
        run_query(config, args.query)
    elif args.generate == "biography":
        run_biography_generator(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
