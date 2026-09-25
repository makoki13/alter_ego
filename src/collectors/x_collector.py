"""
alterEgo - Colector de X (Twitter).
Procesa el archivo exportado desde x.com/settings/download_your_data.
El export es un ZIP que contiene archivos .js con formato JSON.
"""

import json
import zipfile
from pathlib import Path
from typing import Any

from src.collectors.base_collector import BaseCollector


class XCollector(BaseCollector):
    """Recolecta y procesa datos exportados de X/Twitter."""

    def __init__(self, config: dict, raw_dir: Path, processed_dir: Path):
        super().__init__("x_twitter", raw_dir, processed_dir)
        self.archive_path = Path(config.get("archive_path", "data/raw/x_twitter"))

    def collect(self) -> int:
        """
        Verifica que el export está disponible.
        Si es un ZIP, lo extrae. Si ya es carpeta, la usa directamente.
        """
        count = 0

        # Buscar el ZIP o la carpeta extraída
        zip_files = list(self.archive_path.glob("*.zip"))
        extracted_dir = self.archive_path / "data"

        if extracted_dir.exists():
            # Ya está extraído
            print(f"  📂 Export ya extraído en: {extracted_dir}")
            count = len(list(extracted_dir.glob("*.js")))

        elif zip_files:
            # Extraer el primer ZIP encontrado
            zip_file = zip_files[0]
            print(f"  📦 Extrayendo: {zip_file.name}")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(self.archive_path)
            count = len(list((self.archive_path / "data").glob("*.js")))

        else:
            print(f"  ⚠ No se encuentra el export de X en: {self.archive_path}")
            print("     Coloca ahí el .zip descargado de x.com")
            return 0

        print(f"  ✅ Archivos .js encontrados: {count}")
        return count

    def process(self) -> list[dict[str, Any]]:
        """Procesa tweets, perfil y respuestas."""
        records = []
        data_dir = self.archive_path / "data"

        if not data_dir.exists():
            for candidate in self.archive_path.rglob("tweets.js"):
                data_dir = candidate.parent
                break

        if not data_dir.exists():
            print("  ❌ No se encuentra la carpeta 'data' del export.")
            return records

        # Procesar perfil
        profile = self._load_js_file(data_dir / "profile.js")
        if profile:
            profile_record = self._process_profile(profile)
            if profile_record:
                records.append(profile_record)

        # Procesar tweets
        tweets = self._load_js_file(data_dir / "tweets.js")
        if tweets and isinstance(tweets, list):
            records.extend(self._process_tweets(tweets))

        # Procesar respuestas
        replies = self._load_js_file(data_dir / "replies.js")
        if replies and isinstance(replies, list):
            records.extend(self._process_tweets(replies, tweet_type="reply"))

        # Filtrar registros vacíos
        records = [r for r in records if r and r.get("content")]

        print(f"  📄 Registros procesados de X: {len(records)}")
        return records
    # ── Métodos internos ──

    def _load_js_file(self, file_path: Path) -> list | dict | None:
        """
        Carga un archivo .js del export de Twitter.
        Estos archivos tienen formato: window.YTD.xxx.part0 = [...]
        Hay que quitar el prefijo para parsear como JSON.
        """
        if not file_path.exists():
            return None

        try:
            text = file_path.read_text(encoding="utf-8")

            # Quitar el prefijo JavaScript: window.YTD.xxx.part0 =
            if "=" in text:
                json_str = text.split("=", 1)[1].strip()
            else:
                json_str = text

            return json.loads(json_str)

        except (json.JSONDecodeError, IndexError) as e:
            print(f"  ⚠ Error parseando {file_path.name}: {e}")
            return None

    def _process_profile(self, profile_data: list | dict) -> dict[str, Any] | None:
        """Extrae información del perfil."""
        try:
            # El formato puede ser [{...}] o {...}
            if isinstance(profile_data, list) and len(profile_data) > 0:
                profile = profile_data[0].get("profile", profile_data[0])
            elif isinstance(profile_data, dict):
                profile = profile_data.get("profile", profile_data)
            else:
                return None

            content_parts = []
            if profile.get("description"):
                content_parts.append(f"Bio: {profile['description']}")
            if profile.get("location"):
                content_parts.append(f"Ubicación: {profile['location']}")
            if profile.get("name"):
                content_parts.append(f"Nombre: {profile['name']}")
            if profile.get("created_at"):
                content_parts.append(f"Cuenta creada: {profile['created_at']}")

            if not content_parts:
                return None

            return {
                "id": "x_profile_00001",
                "source": "x_twitter",
                "type": "profile",
                "title": "Perfil de X",
                "content": " | ".join(content_parts),
                "date": profile.get("created_at", ""),
                "metadata": {
                    "followers": profile.get("followers_count", ""),
                    "following": profile.get("friends_count", ""),
                    "tweets_count": profile.get("statuses_count", ""),
                },
                "tags": ["perfil", "social"],
            }
        except Exception:
            return None

    def _process_tweets(
        self, tweets_data: list, tweet_type: str = "tweet"
    ) -> list[dict[str, Any]]:
        """Procesa una lista de tweets o respuestas."""
        records = []

        for i, item in enumerate(tweets_data):
            try:
                tweet = item.get("tweet", item)

                text = tweet.get("full_text", "")
                if not text.strip():
                    continue

                # Limpiar URLs de t.co para ahorrar contexto
                # (las dejamos, pero podrían filtrarse)

                record = {
                    "id": f"x_{tweet_type}_{tweet.get('id', i):>020s}",
                    "source": "x_twitter",
                    "type": f"social_{tweet_type}",
                    "title": "",
                    "content": text.strip(),
                    "date": tweet.get("created_at", ""),
                    "metadata": {
                        "likes": tweet.get("favorite_count", "0"),
                        "retweets": tweet.get("retweet_count", "0"),
                        "language": tweet.get("lang", ""),
                    },
                    "tags": ["twitter", tweet_type],
                }
                records.append(record)

            except Exception:
                continue

        return records

