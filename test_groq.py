"""
Verificar que ChatGroq funciona con tu API key.
"""

import os

from dotenv import load_dotenv

load_dotenv()

print(f"\n{'=' * 50}")
print("  Test: ChatGroq")
print(f"{'=' * 50}\n")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("  ❌ GROQ_API_KEY no encontrada en .env")
    exit(1)

print(f"  🔑 API key encontrada: {api_key[:10]}...")

try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        # api_key se lee automáticamente de la variable de entorno GROQ_API_KEY
        temperature=0.7,
        max_tokens=50,
    )

    response = llm.invoke("Di 'Conexión LangChain OK' y nada más.")
    print(f"  ✅ Respuesta del modelo: {response.content}")
    print("\n  🎉 ChatGroq funciona correctamente.\n")

except Exception as e:
    print(f"  ❌ Error: {e}")
