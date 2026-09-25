"""
alterEgo - Colector de Telegram.
Procesa el JSON exportado desde Telegram Desktop.
Formato: Exportar historial del chat → JSON.
"""

import json
from pathlib import Path
from typing import Any

from src.collectors.base_collector import BaseCollector


class TelegramCollector(BaseCollector):
    """Recolecta y procesa mensajes exportados de un grupo de Telegram."""

    def __init__(self, config: dict, raw_dir: Path, processed_dir: Path):
        super().__init__("telegram", raw_dir, processed_dir)
        self.export_path = Path(config.get("export_path", "data/raw/telegram"))
        self.group_name = config.get("group_name", "")

    def collect(self) -> int:
        """
        Verifica que el archivo result.json está en la carpeta.
        Telegram Desktop genera: result.json (y posiblemente fotos/).
        """
        json_files = list(self.export_path.glob("*.json"))

        if not json_files:
            # Buscar en subcarpetas (a veces Telegram crea una subcarpeta)
            json_files = list(self.export_path.rglob("result.json"))

        if not json_files:
            print(f"  ⚠ No se encuentra el export de Telegram en: {self.export_path}")
            print("     Exporta desde Telegram Desktop → Formato JSON")
            return 0

        print(f"  📂 Export encontrado: {json_files[0].name}")
        return len(json_files)

    def process(self) -> list[dict[str, Any]]:
        """Procesa los mensajes del export JSON."""
        records = []

        # Encontrar el archivo JSON
        json_file = self._find_json()
        if not json_file:
            return records

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  ❌ Error leyendo {json_file.name}: {e}")
            return records

        # Extraer metadatos del grupo
        group_name = data.get("name", self.group_name or "Grupo de Telegram")
        messages = data.get("messages", [])

        print(f"  💬 Grupo: {group_name}")
        print(f"  💬 Mensajes totales: {len(messages)}")

        # Procesar mensajes
        for msg in messages:
            record = self._process_message(msg, group_name)
            if record:
                records.append(record)

        # Añadir un registro de contexto del grupo
        records.insert(0, {
            "id": "tg_group_info",
            "source": "telegram",
            "type": "group_info",
            "title": f"Grupo: {group_name}",
            "content": (
                f"Grupo de Telegram llamado '{group_name}'. "
                f"Contiene {len(messages)} mensajes exportados."
            ),
            "date": "",
            "metadata": {"total_messages": len(messages)},
            "tags": ["telegram", "grupo"],
        })

        print(f"  📄 Registros procesados de Telegram: {len(records)}")
        return records

    # ── Métodos internos ──

    def _find_json(self) -> Path | None:
        """Busca el archivo JSON del export."""
        # Buscar result.json directamente
        direct = self.export_path / "result.json"
        if direct.exists():
            return direct

        # Buscar cualquier JSON
        json_files = list(self.export_path.glob("*.json"))
        if json_files:
            return json_files[0]

        # Buscar en subcarpetas
        json_files = list(self.export_path.rglob("result.json"))
        if json_files:
            return json_files[0]

        return None

    def _process_message(
        self, msg: dict, group_name: str
    ) -> dict[str, Any] | None:
        """Convierte un mensaje de Telegram al esquema común."""
        # Solo nos interesan mensajes de texto normales
        if msg.get("type") != "message":
            return None

        # Extraer el texto (puede ser string o lista de entidades)
        text = self._extract_text(msg.get("text", ""))
        if not text or not text.strip():
            return None

        # Ignorar mensajes muy cortos (reacciones, emojis sueltos)
        if len(text.strip()) < 3:
            return None

        sender = msg.get("from", "Desconocido")
        date = msg.get("date", "")

        return {
            "id": f"tg_msg_{msg.get('id', 0):08d}",
            "source": "telegram",
            "type": "chat_message",
            "title": "",
            "content": f"[{sender}]: {text.strip()}",
            "date": date,
            "metadata": {
                "sender": sender,
                "group": group_name,
                "message_id": msg.get("id"),
            },
            "tags": ["telegram", "chat", group_name.lower().replace(" ", "_")],
        }

    def _extract_text(self, text_field: str | list) -> str:
        """
        Extrae texto del campo 'text' de Telegram.
        Puede ser un string simple o una lista de entidades.
        Ejemplo de lista: ["Hola ", {"type": "bold", "text": "mundo"}]
        """
        if isinstance(text_field, str):
            return text_field

        if isinstance(text_field, list):
            parts = []
            for item in text_field:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("text", ""))
            return "".join(parts)

        return str(text_field)
