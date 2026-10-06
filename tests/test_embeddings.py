"""
Verificar que sentence-transformers genera embeddings.
⚠ La primera vez descarga el modelo (~470 MB). Puede tardar.
"""

print(f"\n{'=' * 50}")
print("  Test: Sentence-Transformers")
print(f"{'=' * 50}\n")

try:
    from sentence_transformers import SentenceTransformer

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print("  🧠 Cargando modelo: {model_name}")
    print("     (La primera vez descarga ~470 MB, puede tardar)")

    model = SentenceTransformer(model_name)

    # Generar embedding de prueba
    test_text = "Me llamo Pablo y me gusta el ciclismo."
    embedding = model.encode([test_text])

    print("  ✅ Embedding generado.")
    print(f"     Dimensión: {embedding.shape[1]}")
    print(f"     Primeros 5 valores: {embedding[0][:5].tolist()}")
    print("\n  🎉 Sentence-Transformers OK.\n")

except Exception as e:
    print(f"  ❌ Error: {e}")
