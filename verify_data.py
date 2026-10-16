"""
alterEgo - Verificación de datos tras la recolección.
Uso: python verify_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

def main():
    print(f"\n{'=' * 50}")
    print("  alterEgo - Verificación de datos")
    print(f"{'=' * 50}\n")

    processed_dir = Path("data/processed")

    if not processed_dir.exists():
        print("  ❌ No existe data/processed/")
        return

    total = 0
    sources = {}

    for json_file in sorted(processed_dir.glob("*_records.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                records = json.load(f)

            count = len(records)
            total += count

            # Contar por fuente
            for rec in records:
                src = rec.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1

            # Mostrar primeros registros como muestra
            print(f"  📂 {json_file.name}: {count} registros")

            if records:
                sample = records[0]
                print(f"     Ejemplo: [{sample.get('type', '?')}] {str(sample.get('content', ''))[:80]}...")

        except Exception as e:
            print(f"  ❌ Error en {json_file.name}: {e}")

    print(f"\n{'─' * 50}")
    print(f"  📊 Total de registros: {total}")
    print("\n  Por fuente:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}")

    # Verificar índice FAISS
    vectors_dir = Path("data/vectors")
    index_file = vectors_dir / "index.faiss"
    if index_file.exists():
        print(f"\n  ✅ Índice FAISS existe en: {vectors_dir}")
    else:
        print("\n  ⚠ No hay índice FAISS. Ejecuta: python main.py --build-index")

    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()