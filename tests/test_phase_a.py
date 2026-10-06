"""
alterEgo - FASE A: Verificación de dependencias.
Ejecuta: python test_phase_a.py
"""

import sys


def check(name: str, import_fn):
    try:
        import_fn()
        print(f"  ✅ {name}")
        return True
    except ImportError as e:
        print(f"  ❌ {name}: {e}")
        return False

print("\n🎭 alterEgo - FASE A: Verificación de dependencias")
print(f"   Python: {sys.version}")
print(f"{'=' * 50}\n")

results = []

# LangChain core
results.append(check("langchain", lambda: __import__("langchain")))
results.append(check("langchain-core", lambda: __import__("langchain_core")))
results.append(check("langchain-community", lambda: __import__("langchain_community")))
results.append(check("langchain-groq", lambda: __import__("langchain_groq")))

# Agentes
results.append(check("langgraph", lambda: __import__("langgraph")))

# Embeddings
results.append(check("sentence-transformers", lambda: __import__("sentence_transformers")))

# Vector store
results.append(check("faiss-cpu", lambda: __import__("faiss")))

# Utilidades
results.append(check("numpy", lambda: __import__("numpy")))
results.append(check("pyyaml", lambda: __import__("yaml")))
results.append(check("python-dotenv", lambda: __import__("dotenv")))

# Lectura de documentos
results.append(check("Pillow", lambda: __import__("PIL")))
results.append(check("python-docx", lambda: __import__("docx")))
results.append(check("PyPDF2", lambda: __import__("PyPDF2")))

print(f"\n{'=' * 50}")
passed = sum(results)
total = len(results)
print(f"  Resultado: {passed}/{total} OK")

if passed == total:
    print("  🎉 Todo instalado correctamente.")
else:
    print("  ⚠ Hay dependencias pendientes. Revisa los ❌ de arriba.")

print()
