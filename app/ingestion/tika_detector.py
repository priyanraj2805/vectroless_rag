from pathlib import Path

EXTENSION_MIME = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":  "application/vnd.ms-powerpoint",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
    ".txt":  "text/plain",
    ".html": "text/html",
    ".htm":  "text/html",
    ".rtf":  "application/rtf",
}


class TikaDetector:
    def __init__(self):
        self._available = False
        try:
            import tika
            tika.initVM()
            from tika import parser as tika_parser
            self._tika_parser = tika_parser
            self._available = True
            print("[tika] Java server ready — using magic-byte detection")
        except Exception as e:
            print(f"[tika] Not available ({e}) — using extension fallback")

    def detect(self, file_path: str) -> dict:
        """
        Returns:
          {
            "mime_type": "application/pdf",
            "metadata": {
              "author": "Jane Doe" | None,
              "title": "Q3 Report" | None,
              "language": "en" | None,
              "page_count": 12 | None,
              "created": "2024-01-15" | None,
            }
          }
        """
        path = Path(file_path)

        if self._available:
            try:
                result = self._tika_parser.from_file(
                    str(file_path),
                    requestOptions={"timeout": 30},
                )
                raw_meta = result.get("metadata") or {}

                mime = self._first(raw_meta, ["Content-Type", "content-type"])
                if isinstance(mime, str):
                    mime = mime.split(";")[0].strip()
                mime = mime or self._ext_mime(path)

                return {
                    "mime_type": mime,
                    "metadata": {
                        "author":     self._first(raw_meta, ["Author", "meta:author", "creator", "dc:creator"]),
                        "title":      self._first(raw_meta, ["dc:title", "title", "pdf:docinfo:title"]),
                        "language":   self._first(raw_meta, ["language", "dc:language", "Content-Language"]),
                        "page_count": self._to_int(self._first(raw_meta, ["xmpTPg:NPages", "Page-Count", "meta:page-count", "pdf:docinfo:page-count"])),
                        "created":    self._first(raw_meta, ["dcterms:created", "meta:creation-date", "created", "Creation-Date"]),
                    },
                }
            except Exception as e:
                print(f"[tika] Detection error for {path.name}: {e} — falling back to extension")

        return {
            "mime_type": self._ext_mime(path),
            "metadata": {
                "author": None, "title": None,
                "language": None, "page_count": None, "created": None,
            },
        }

    def _ext_mime(self, path: Path) -> str:
        return EXTENSION_MIME.get(path.suffix.lower(), "application/octet-stream")

    def _first(self, d: dict, keys: list):
        for k in keys:
            v = d.get(k)
            if v:
                return v[0] if isinstance(v, list) else v
        return None

    def _to_int(self, v) -> int | None:
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None
