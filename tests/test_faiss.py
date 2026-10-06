"""
Verificar que FAISS funciona correctamente.
"""

import numpy as np

print(f"\n{'=' * 50}")
print("  Test: FAISS")
print(f"{'=' * 50}\n")

try:
    import faiss

    # Crear un índice pequeño de prueba
    dimension = 384
    index = faiss.IndexFlatIP(dimension)

    # Añadir 5 vectores aleatorios
    vectors = np.random.rand(5, dimension).astype(np.float32)
    index.add(vectors)

    # Buscar
    query = np.random.rand(1, dimension).astype(np.float32)
    scores, indices = index.search(query, 3)

    print("  ✅ FAISS funciona.")
    print(f"     Índice: {index.ntotal} vectores")
    print(f"     Búsqueda: top-3 → índices {indices[0].tolist()}")
    print("\n  🎉 FAISS OK.\n")

except Exception as e:
    print(f"  ❌ Error: {e}")
