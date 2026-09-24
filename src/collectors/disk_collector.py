"""
alterEgo - Colector de disco duro.
Escanea carpetas locales y extrae texto o metadatos según el tipo.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.collectors.base_collector import BaseCollector


class DiskCollector(BaseCollector):
    """Recolecta archivos del disco duro local (textos e imágenes)."""

    def __init__(self, config: dict, raw_dir: Path, processed_dir: Path):
        super().__init__("disk", raw_dir, processed_dir)
        self.sources = config.get("paths", [])
        self.max_depth = config.get("max_depth", 5)
        self.max_file_size = config.get("max_file_size_mb", 50) * 1024 * 1024

    # ── COLLECT ──

    def collect(self) -> int:
        total = 0

        for source in self.sources:
            base_path = Path(source["path"])
            source_type = source["type"]
            extensions = set(source.get("extensions", []))

            if not base_path.exists():
                print(f"  ⚠ Ruta no encontrada: {base_path}")
                continue

            print(f"  📂 Escaneando: {base_path}  (tipo: {source_type})")
            count = 0

            for file_path in self._walk(base_path, depth=0):
                if self._is_valid(file_path, extensions):
                    self._copy_to_raw(file_path, source_type)
                    count += 1

            print(f"     → {count} archivos de tipo '{source_type}'")
            total += count

        print(f"  ✅ Total recolectados: {total}")
        return total

    # ── PROCESS ──

    def process(self) -> list[dict[str, Any]]:
        records = []

        for source_type in ("documents", "images"):
            type_dir = self.raw_dir / source_type
            if not type_dir.exists():
                continue

            for file_path in sorted(type_dir.iterdir()):
                if not file_path.is_file():
                    continue

                if source_type == "images":
                    record = self._process_image(file_path)
                else:
                    record = self._process_document(file_path)

                if record and record.get("content"):
                    records.append(record)

        print(f"  📄 Registros procesados: {len(records)}")
        return records

    # ── DOCUMENTOS ──

    def _process_document(self, file_path: Path) -> dict[str, Any] | None:
        content = self._read_text_file(file_path)
        if not content or not content.strip():
            return None

        return {
            "id": f"disk_doc_{file_path.stem}_{abs(hash(str(file_path))) % 100000:05d}",
            "source": "disk",
            "type": "document",
            "title": file_path.stem,
            "content": content.strip(),
            "date": self._get_date(file_path),
            "metadata": {
                "original_path": str(file_path),
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
            },
            "tags": [],
        }

    # ── IMÁGENES ──

    def _process_image(self, file_path: Path) -> dict[str, Any] | None:
        metadata = {
            "original_path": str(file_path),
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
        }

        exif_data = self._extract_exif(file_path)
        if exif_data:
            metadata["exif"] = exif_data

        content_parts = [f"Imagen: {file_path.name}"]

        if exif_data.get("date_taken"):
            content_parts.append(f"Fecha de captura: {exif_data['date_taken']}")
        if exif_data.get("camera"):
            content_parts.append(f"Cámara: {exif_data['camera']}")
        if exif_data.get("location"):
            content_parts.append(f"Ubicación: {exif_data['location']}")
        if exif_data.get("width") and exif_data.get("height"):
            content_parts.append(
                f"Resolución: {exif_data['width']}x{exif_data['height']}"
            )

        name_hint = file_path.stem.replace("_", " ").replace("-", " ")
        content_parts.append(f"Nombre descriptivo: {name_hint}")

        content = " | ".join(content_parts)

        return {
            "id": f"disk_img_{file_path.stem}_{abs(hash(str(file_path))) % 100000:05d}",
            "source": "disk",
            "type": "image_metadata",
            "title": file_path.stem,
            "content": content,
            "date": exif_data.get("date_taken", self._get_date(file_path)),
            "metadata": metadata,
            "tags": [],
        }

    def _extract_exif(self, file_path: Path) -> dict:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(file_path)
            exif_raw = img.getexif()  # ← API pública, sin guión bajo

            if not exif_raw:
                return {"width": img.width, "height": img.height}

            exif = {}
            for tag_id, value in exif_raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal":
                    exif["date_taken"] = str(value)
                elif tag == "Model":
                    exif["camera"] = str(value)
                elif tag == "GPSInfo":
                    exif["location"] = self._parse_gps(value)

            exif["width"] = img.width
            exif["height"] = img.height
            return exif

        except ImportError:
            print("  ⚠ Instala Pillow: pip install Pillow")
            return {}
        except Exception:
            return {}

    def _parse_gps(self, gps_info: dict) -> str:
        try:
            from PIL.ExifTags import GPSTAGS

            gps = {}
            for key, val in gps_info.items():
                tag = GPSTAGS.get(key, key)
                gps[tag] = val

            lat = gps.get("GPSLatitude")
            lon = gps.get("GPSLongitude")

            if lat and lon:
                lat_deg = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / 3600
                lon_deg = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / 3600

                if gps.get("GPSLatitudeRef") == "S":
                    lat_deg = -lat_deg
                if gps.get("GPSLongitudeRef") == "W":
                    lon_deg = -lon_deg

                return f"{lat_deg:.6f}, {lon_deg:.6f}"
            return ""
        except Exception:
            return ""

    # ── UTILIDADES ──

    def _walk(self, path: Path, depth: int):
        if depth > self.max_depth:
            return
        try:
            for entry in path.iterdir():
                if entry.is_file():
                    yield entry
                elif entry.is_dir():
                    yield from self._walk(entry, depth + 1)
        except PermissionError:
            print(f"  🔒 Sin permiso: {path}")

    def _is_valid(self, file_path: Path, extensions: set) -> bool:
        if file_path.suffix.lower() not in extensions:
            return False
        try:
            if file_path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
        return True

    def _copy_to_raw(self, file_path: Path, source_type: str):
        dest_dir = self.raw_dir / source_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / file_path.name

        if dest.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(file_path, dest)

    def _read_text_file(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        try:
            if suffix in (".txt", ".md", ".csv", ".log"):
                return file_path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".docx":
                return self._read_docx(file_path)
            elif suffix == ".pdf":
                return self._read_pdf(file_path)
        except Exception as e:
            print(f"  ❌ Error leyendo {file_path.name}: {e}")
        return ""

    def _read_docx(self, file_path: Path) -> str:
        try:
            from docx import Document
            doc = Document(str(file_path))  # ← str(file_path) en vez de file_path
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print("  ⚠ pip install python-docx")
            return ""

    def _read_pdf(self, file_path: Path) -> str:
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("  ⚠ pip install PyPDF2")
            return ""

    def _get_date(self, file_path: Path) -> str:
        try:
            return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        except OSError:
            return ""
