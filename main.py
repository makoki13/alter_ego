"""
alterEgo - Punto de entrada principal.
Uso: python main.py --collect disk
"""

import sys
from pathlib import Path

# Asegurar que Python encuentra el paquete 'src'
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import yaml


def load_config() -> dict:
    """Carga la configuración desde config/settings.yaml."""
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

    # Fase 1: Recolectar
    print("\n[1/2] Recolectando archivos...")
    count = collector.collect()

    if count == 0:
        print("⚠ No se encontraron archivos. Revisa las rutas en settings.yaml.")
        return

    # Fase 2: Procesar
    print("\n[2/2] Procesando archivos...")
    records = collector.process()

    if not records:
        print("⚠ Se recolectaron archivos pero no se pudo procesar ninguno.")
        return

    # Guardar resultado
    output_file = processed_dir / "disk_records.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado en: {output_file}")
    print(f"   Registros: {len(records)}")

    # Vista previa del primer registro
    print("\n--- Primer registro ---")
    preview = records[0].copy()
    if len(preview.get("content", "")) > 200:
        preview["content"] = preview["content"][:200] + "..."
    print(json.dumps(preview, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="alterEgo - Tu gemelo digital")
    parser.add_argument(
        "--collect",
        type=str,
        choices=["disk", "gdrive", "x_twitter", "telegram", "all"],
        default="disk",
        help="Fuente a recolectar",
    )
    args = parser.parse_args()

    config = load_config()
    print(f"\n🎭 alterEgo v{config['project']['version']}")

    if args.collect == "disk":
        run_disk_collector(config)
    elif args.collect == "all":
        run_disk_collector(config)
        # Aquí se añadirán los demás colectores cuando estén implementados
    else:
        print(f"⚠ Colector '{args.collect}' aún no implementado. Próximamente.")


if __name__ == "__main__":
    main()
