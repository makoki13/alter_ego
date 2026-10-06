"""
alterEgo - Herramienta: obtener información del perfil.
"""

from langchain_core.tools import tool


def create_profile_tool(profile_data: dict):
    """Crea la herramienta de perfil."""

    @tool
    def get_profile() -> str:
        """Devuelve información básica del perfil del usuario:
        nombre, profesión, tono, aficiones."""
        parts = []
        if profile_data.get("name"):
            parts.append(f"Nombre: {profile_data['name']}")
        if profile_data.get("profession"):
            parts.append(f"Profesión: {profile_data['profession']}")
        if profile_data.get("tone"):
            parts.append(f"Tono: {profile_data['tone']}")
        if profile_data.get("interests"):
            parts.append(f"Aficiones: {', '.join(profile_data['interests'])}")
        return "\n".join(parts)

    return get_profile
