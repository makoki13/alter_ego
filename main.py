"""
https://console.groq.com/keys

python -m venv .venv
.venv bckslash Scripts\activate.bat # type: ignore

alterEgo - Punto de entrada principal.
Uso:
  python main.py --collect disk
  python main.py --generate biography
"""  # noqa: W605

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


def run_disk_collector(config: dict):
    """Ejecuta el colector de disco duro."""
    from src.collectors.disk_collector import DiskCollector

    source_cfg = config["sources"]["disk"]
    if not source_cfg.get("enabled", False):
        print("❌ Colector de disco desactivado en settings.yaml")
        return

    raw_dir = Path("data/raw/disk")
    processed_dir = Path("data/processed")

    collector = DiskCollector(source_cfg, raw_dir, processed_dir)

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
        print("⚠ No se pudo procesar ningún archivo.")
        return

    output_file = processed_dir / "disk_records.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado en: {output_file}")
    print(f"   Registros: {len(records)}")


def run_biography_generator(config: dict):
    """Genera la biografía usando los datos procesados."""
    from src.generators.biography import BiographyGenerator
    from src.llm.factory import LLMFactory

    # Cargar registros procesados
    records_file = Path("data/processed/disk_records.json")
    if not records_file.exists():
        print("❌ No hay datos procesados. Ejecuta primero: python main.py --collect disk")
        return

    with open(records_file, encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("❌ El archivo de registros está vacío.")
        return

    print(f"\n{'=' * 50}")
    print("  alterEgo - Generador de biografía")
    print(f"{'=' * 50}")

    # Inicializar LLM
    llm_config = config.get("llm", {})
    llm = LLMFactory.get(llm_config)
    print(f"  🔌 Proveedor: {llm.get_model_name()}")

    # Generar
    generator = BiographyGenerator(llm, config)
    biography = generator.generate(records)

    # Vista previa
    print(f"\n{'=' * 50}")
    print("  Vista previa (primeros 500 chars):")
    print(f"{'=' * 50}")
    print(biography[:500])
    print("...")


def main():
    parser = argparse.ArgumentParser(description="alterEgo - Tu gemelo digital")
    parser.add_argument(
        "--collect",
        type=str,
        choices=["disk", "gdrive", "x_twitter", "telegram", "all"],
        help="Fuente a recolectar",
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
    elif args.collect == "all":
        run_disk_collector(config)
    elif args.generate == "biography":
        run_biography_generator(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
