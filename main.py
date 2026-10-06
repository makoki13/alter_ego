"""
alterEgo - Punto de entrada principal (LangChain edition).
Uso:
  python main.py --collect disk
  python main.py --collect all
  python main.py --build-index
  python main.py --generate biography
  python main.py --chat
  python main.py --query "¿Dónde veraneaba?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import yaml


def load_config() -> dict:
    with open("config/settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_records() -> list[dict]:
    """Combina todos los registros procesados."""
    processed_dir = Path("data/processed")
    all_records = []
    for json_file in sorted(processed_dir.glob("*_records.json")):
        with open(json_file, encoding="utf-8") as f:
            records = json.load(f)
        print(f"  📂 {json_file.name}: {len(records)} registros")
        all_records.extend(records)
    print(f"  📚 Total: {len(all_records)} registros\n")
    return all_records


# ── COLECTORES (sin cambios, usan nuestro código) ──

def run_collect(config: dict, source: str):
    """Ejecuta colectores."""
    if source in ("disk", "all"):
        from src.collectors.disk_collector import DiskCollector
        cfg = config["sources"]["disk"]
        collector = DiskCollector(cfg, Path("data/raw/disk"), Path("data/processed"))
        print("\n  📂 Recolectando disco...")
        count = collector.collect()
        records = collector.process()
        with open("data/processed/disk_records.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  ✅ {len(records)} registros\n")

    if source in ("x_twitter", "all"):
        from src.collectors.x_collector import XCollector
        cfg = config["sources"]["x_twitter"]
        collector = XCollector(cfg, Path("data/raw/x_twitter"), Path("data/processed"))
        print("\n  🐦 Recolectando X/Twitter...")
        count = collector.collect()
        if count > 0:
            records = collector.process()
            with open("data/processed/x_records.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {len(records)} registros\n")

    if source in ("telegram", "all"):
        from src.collectors.telegram_collector import TelegramCollector
        cfg = config["sources"]["telegram"]
        collector = TelegramCollector(cfg, Path("data/raw/telegram"), Path("data/processed"))
        print("\n  💬 Recolectando Telegram...")
        count = collector.collect()
        if count > 0:
            records = collector.process()
            with open("data/processed/telegram_records.json", "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {len(records)} registros\n")


# ── ÍNDICE VECTORIAL ──

def run_build_index(config: dict):
    """Construye el índice FAISS usando LangChain."""
    from src.memory.vector_memory import VectorMemory

    print(f"\n{'=' * 50}")
    print("  alterEgo - Construir índice vectorial")
    print(f"{'=' * 50}\n")

    records = load_all_records()
    if not records:
        print("❌ No hay registros. Ejecuta: python main.py --collect all")
        return

    memory = VectorMemory(config)
    memory.build_from_records(records)
    print("\n✅ Índice construido.")


# ── GENERAR BIOGRAFÍA ──

def run_biography(config: dict):
    """Genera biografía usando el agente LangChain."""
    from harness.callbacks import AlterEgoCallback
    from src.agent.alter_ego_agent import AlterEgoAgent

    print(f"\n{'=' * 50}")
    print("  alterEgo - Generación de biografía")
    print(f"{'=' * 50}\n")

    callback = AlterEgoCallback()
    agent = AlterEgoAgent(config)
    agent.setup()

    # Generar
    print("  🤖 Generando biografía...\n")
    biography = agent.generate_biography()

    # Guardar
    output_path = Path("output/biografia.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(biography)

    print(f"\n  💾 Guardado en: {output_path}")
    print(f"\n  📊 {callback.summary()}")
    print(f"\n{'=' * 50}")
    print(biography[:500])
    print("...")


# ── CHAT ──

def run_chat(config: dict):
    """Chat interactivo con alterEgo."""
    from src.agent.alter_ego_agent import AlterEgoAgent

    print(f"\n{'=' * 50}")
    print("  🎭 alterEgo - Chat")
    print("  Escribe 'salir' para terminar.")
    print(f"{'=' * 50}\n")

    agent = AlterEgoAgent(config)
    agent.setup()

    chat_history = []

    while True:
        user_input = input("\n  Tú: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            print("\n  👋 Hasta pronto.")
            break

        response = agent.chat(user_input, chat_history)
        print(f"\n  alterEgo: {response}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})


# ── QUERY ──

def run_query(config: dict, query: str):
    """Consulta directa al índice vectorial."""
    from src.memory.vector_memory import VectorMemory

    memory = VectorMemory(config)
    if not memory.load():
        print("❌ No hay índice. Ejecuta: python main.py --build-index")
        return

    results = memory.search(query, top_k=5)
    print(f"\n  Consulta: \"{query}\"")
    print(f"  Resultados: {len(results)}\n")

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "?")
        date = doc.metadata.get("date", "")[:10]
        print(f"  [{i}] [{source}] [{date}] {doc.page_content[:150]}...")
        print()


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser(description="🎭 alterEgo")
    parser.add_argument("--collect", choices=["disk", "x_twitter", "telegram", "all"])
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--generate", choices=["biography"])
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--query", type=str)
    args = parser.parse_args()

    config = load_config()
    print(f"\n🎭 alterEgo v{config['project']['version']} (LangChain)")

    if args.collect:
        run_collect(config, args.collect)
    elif args.build_index:
        run_build_index(config)
    elif args.generate == "biography":
        run_biography(config)
    elif args.chat:
        run_chat(config)
    elif args.query:
        run_query(config, args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
