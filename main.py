"""
alter-ego - Punto de entrada principal.
Uso:
  python main.py --transcribe          # Transcribir audios nuevos
  python main.py --store               # Indexar transcripciones en FAISS
  python main.py --process-audio       # Transcribir + indexar en un paso
  python main.py --biography           # Generar biografía
  python main.py --assistant           # Asistente personal (chat)
  python main.py --query "pregunta"    # Consulta directa al índice
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import yaml


def load_config() -> dict:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        print("❌ No se encuentra config/settings.yaml")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── FASE 2: Transcripción ──

def run_transcribe(config: dict):
    """Transcribe todos los audios nuevos en data/audio/."""
    # Se implementará en Fase 2
    print("\n  ⬜ Transcripción: pendiente de implementar (Fase 2)")


# ── FASE 3: Almacenamiento ──

def run_store(config: dict):
    """Indexa todas las transcripciones en FAISS."""
    # Se implementará en Fase 3
    print("\n  ⬜ Almacenamiento: pendiente de implementar (Fase 3)")


def run_process_audio(config: dict):
    """Pipeline completo: transcribir + almacenar."""
    run_transcribe(config)
    run_store(config)


# ── FASE 5: Biografía ──

def run_biography(config: dict):
    """Genera la biografía a partir de las transcripciones."""
    # Se implementará en Fase 5
    print("\n  ⬜ Biografía: pendiente de implementar (Fase 5)")


# ── FASE 6: Asistente ──

def run_assistant(config: dict):
    """Chat interactivo con el asistente personal."""
    # Se implementará en Fase 6
    print("\n  ⬜ Asistente: pendiente de implementar (Fase 6)")


# ── Query ──

def run_query(config: dict, query: str):
    """Consulta directa al índice vectorial."""
    # Se implementará en Fase 3
    print("\n  ⬜ Query: pendiente de implementar (Fase 3)")


# ── MAIN ──

def main():
    parser = argparse.ArgumentParser(description="🎭 alter-ego")
    parser.add_argument("--transcribe", action="store_true",
                        help="Transcribir audios nuevos")
    parser.add_argument("--store", action="store_true",
                        help="Indexar transcripciones en FAISS")
    parser.add_argument("--process-audio", action="store_true",
                        help="Transcribir + indexar en un paso")
    parser.add_argument("--biography", action="store_true",
                        help="Generar biografía")
    parser.add_argument("--assistant", action="store_true",
                        help="Asistente personal (chat)")
    parser.add_argument("--query", type=str,
                        help="Consulta directa al índice")
    args = parser.parse_args()

    config = load_config()
    print(f"\n🎭 {config['project']['name']} v{config['project']['version']}")

    if args.transcribe:
        run_transcribe(config)
    elif args.store:
        run_store(config)
    elif args.process_audio:
        run_process_audio(config)
    elif args.biography:
        run_biography(config)
    elif args.assistant:
        run_assistant(config)
    elif args.query:
        run_query(config, args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()