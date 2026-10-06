"""
alterEgo - Herramienta: obtener información del perfil.
"""

from langchain_core.tools import tool


def create_profile_tool(profile_data: dict):
    """Crea la herramienta de perfil."""

    @tool
    def get_profile() -> str:
        """Devuelve información básica del perfil del usuario:
        nombre, profesión, tono, aficiones.
        Úsala siempre al inicio para saber quién eres."""
        parts = []
        if profile_data.get("name"):
            parts.append(f"Nombre: {profile_data['name']}")
        if profile_data.get("profession"):
            parts.append(f"Profesión: {profile_data['profession']}")
        if profile_data.get("tone"):
            parts.append(f"Tono: {profile_data['tone']}")
        if profile_data.get("interests"):
            interests = profile_data["interests"]
            if isinstance(interests, list):
                parts.append(f"Aficiones: {', '.join(interests)}")
            else:
                parts.append(f"Aficiones: {interests}")
        if profile_data.get("boundaries"):
            parts.append(f"Temas a evitar: {', '.join(profile_data['boundaries'])}")

        if not parts:
            return "No hay información de perfil disponible."
        return "\n".join(parts)

    return get_profile
