"""
alterEgo - Test de herramientas (Fase C).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml

from src.memory.vector_memory import VectorMemory
from src.tools.get_profile import create_profile_tool
from src.tools.search_by_date import create_date_search_tool
from src.tools.search_memory import create_search_tool

# Cargar config
with open("config/settings.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Cargar perfil
with open("config/user_profile.yaml", encoding="utf-8") as f:
    profile = yaml.safe_load(f)

print(f"\n{'=' * 50}")
print("  alterEgo - Test de herramientas")
print(f"{'=' * 50}\n")

# Cargar memoria
memory = VectorMemory(config)
if not memory.load():
    print("❌ No hay índice. Ejecuta: python main.py --build-index")
    sys.exit(1)

# Crear herramientas
search_tool = create_search_tool(memory)
profile_tool = create_profile_tool(profile)
date_tool = create_date_search_tool(memory)

# Test 1: Perfil
print("  ── Test: get_profile ──")
result = profile_tool.invoke({})
print(f"  {result}\n")

# Test 2: Búsqueda libre
print("  ── Test: search_memory ──")
result = search_tool.invoke({"query": "ciclismo"})
print(f"  {result[:300]}...\n")

# Test 3: Búsqueda por fecha
print("  ── Test: search_by_date ──")
result = date_tool.invoke({"year": "2024", "query": "trabajo"})
print(f"  {result[:300]}...\n")

print(f"{'=' * 50}")
print("  ✅ Herramientas funcionando correctamente.")
print(f"{'=' * 50}\n")
