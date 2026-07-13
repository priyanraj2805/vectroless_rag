MIME_TO_CATEGORY = {
    "application/pdf":                                                                          "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":                 "docx",
    "application/msword":                                                                       "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":                       "xlsx",
    "application/vnd.ms-excel":                                                                "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":               "pptx",
    "application/vnd.ms-powerpoint":                                                           "pptx",
    "image/png":   "image",
    "image/jpeg":  "image",
    "image/jpg":   "image",
    "image/tiff":  "image",
    "image/bmp":   "image",
    "image/webp":  "image",
    "image/gif":   "image",
    "text/plain":  "text",
    "text/html":   "text",
    "text/htm":    "text",
    "application/rtf": "text",
}

# Categories that Docling can handle with structure-aware parsing
DOCLING_CATEGORIES = {"pdf", "docx", "xlsx", "pptx", "image"}


class DocumentRouter:
    def route(self, mime_type: str) -> dict:
        """
        Returns:
          {
            "supported": True | False,
            "category": "pdf" | "docx" | "xlsx" | "pptx" | "image" | "text" | "unknown",
            "use_docling": True | False,
            "use_ocr": True | False,
          }
        """
        category = MIME_TO_CATEGORY.get(mime_type)

        if not category:
            prefix = mime_type.split("/")[0] if "/" in mime_type else ""
            if prefix == "image":
                category = "image"
            elif prefix == "text":
                category = "text"

        if not category:
            return {"supported": False, "category": "unknown", "use_docling": False, "use_ocr": False}

        return {
            "supported": True,
            "category": category,
            "use_docling": category in DOCLING_CATEGORIES,
            "use_ocr": category == "image",
        }
