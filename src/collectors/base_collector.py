"""
alterEgo - Colector base.
Todos los colectores heredan de esta clase.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseCollector(ABC):
    """Interfaz común para todos los colectores de datos."""

    def __init__(self, name: str, raw_dir: Path, processed_dir: Path):
        self.name = name
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def collect(self) -> int:
        pass

    @abstractmethod
    def process(self) -> list[dict[str, Any]]:
        pass

    @staticmethod
    def get_schema() -> dict[str, str]:
        return {
            "id": "str",
            "source": "str",
            "type": "str",
            "title": "str",
            "content": "str",
            "date": "str",
            "metadata": "dict",
            "tags": "list",
        }
