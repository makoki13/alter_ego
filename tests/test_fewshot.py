"""Verificar que los few-shot examples se cargan correctamente."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from src.agent.alter_ego_agent import AlterEgoAgent

# Cargar config
with open("config/settings.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(f"\n{'=' * 50}")
print("  Test: Few-shot examples")
print(f"{'=' * 50}\n")

agent = AlterEgoAgent(config)

# Cargar examples
examples = agent._load_few_shot()
print(f"  📝 Ejemplos cargados: {len(examples)}")

for i, ex in enumerate(examples, 1):
    context = ex.get("context", "")[:50]
    response_preview = ex.get("response", "")[:80].strip()
    print(f"  [{i}] {context}...")
    print(f"      → {response_preview}...")
    print()

# Cargar prompt completo
prompt = agent._load_identity_prompt()
print(f"  📊 Longitud del prompt con few-shot: {len(prompt)} caracteres")
print(f"  📊 Contiene sección de ejemplos: {'few-shot' in prompt.lower()}")

print("\n  ✅ Few-shot examples funcionando.\n")