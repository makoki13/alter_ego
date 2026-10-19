"""
alter-ego - Colector de audio.
Transcribe ficheros m4a a texto usando Groq Whisper.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class AudioCollector:
    """Recolecta audios m4a y los transcribe a texto."""

    def __init__(self, config: dict):
        audio_config = config.get("audio", {})
        self.input_dir = Path(audio_config.get("input_dir", "data/audio"))
        self.output_dir = Path(audio_config.get("output_dir", "data/transcriptions"))
        self.supported_formats = audio_config.get(
            "supported_formats", [".m4a", ".mp3", ".wav", ".ogg"]
        )
        self.whisper_model = audio_config.get("whisper_model", "whisper-large-v3-turbo")
        self.language = audio_config.get("language", "es")

        # Asegurar carpetas
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cliente Groq
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Registro de archivos procesados
        self.manifest_path = self.output_dir / "_manifest.json"
        self.manifest: dict[str, dict] = self._load_manifest()

    def collect(self) -> list[Path]:
        """
        Busca archivos de audio nuevos (no procesados aún).
        Devuelve la lista de rutas a transcribir.
        """
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(self.input_dir.glob(f"*{ext}"))

        # Filtrar solo los nuevos
        new_files = []
        for f in sorted(audio_files):
            if f.name not in self.manifest:
                new_files.append(f)

        print(f"  🎵 Audios en carpeta: {len(audio_files)}")
        print(f"  🆕 Nuevos por transcribir: {len(new_files)}")
        print(f"  ✅ Ya transcritos: {len(audio_files) - len(new_files)}")

        return new_files

    def transcribe(self, audio_path: Path) -> dict[str, Any] | None:
        """
        Transcribe un archivo de audio usando Groq Whisper.
        Devuelve el registro de transcripción.
        """
        print(f"  🎙 Transcribiendo: {audio_path.name}")

        # Verificar tamaño (límite 25 MB de Groq)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            print(f"    ⚠ Archivo demasiado grande ({file_size_mb:.1f} MB > 25 MB). Saltando.")
            return None

        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, file.read()),
                    model=self.whisper_model,
                    language=self.language,
                    response_format="verbose_json",
                )

            # Extraer texto y metadatos
            text = transcription.text.strip()
            duration = getattr(transcription, "duration", 0.0)

            if not text:
                print("    ⚠ Transcripción vacía. Saltando.")
                return None

            # Construir registro
            now = datetime.now()
            record = {
                "id": f"audio_{now:%Y%m%d_%H%M%S}_{audio_path.stem}",
                "filename": audio_path.name,
                "date_transcribed": now.isoformat(),
                "duration_seconds": round(duration, 1),
                "word_count": len(text.split()),
                "transcription": text,
                "status": "completed",
            }

            print(f"    ✅ {record['word_count']} palabras, {duration:.0f}s")
            return record

        except Exception as e:
            print(f"    ❌ Error transcribiendo {audio_path.name}: {e}")
            return None

    def transcribe_all(self) -> list[dict[str, Any]]:
        """Transcribe todos los audios nuevos."""
        new_files = self.collect()

        if not new_files:
            print("\n  ℹ No hay audios nuevos para transcribir.")
            return []

        records = []
        for i, audio_path in enumerate(new_files, 1):
            print(f"\n  [{i}/{len(new_files)}]", end="")
            record = self.transcribe(audio_path)
            if record:
                self._save_record(record)
                self._update_manifest(audio_path.name, record)
                records.append(record)

        self._save_manifest()

        print(f"\n  📊 Transcritos en esta sesión: {len(records)}")
        return records

    def _save_record(self, record: dict[str, Any]):
        """Guarda la transcripción como JSON individual."""
        output_file = self.output_dir / f"{record['id']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def _update_manifest(self, filename: str, record: dict[str, Any]):
        """Actualiza el registro de archivos procesados."""
        self.manifest[filename] = {
            "id": record["id"],
            "date_transcribed": record["date_transcribed"],
            "word_count": record["word_count"],
            "duration_seconds": record["duration_seconds"],
        }

    def _load_manifest(self) -> dict[str, dict]:
        """Carga el manifiesto de archivos ya procesados."""
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Guarda el manifiesto."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)