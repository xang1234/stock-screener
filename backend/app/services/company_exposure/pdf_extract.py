"""Standalone text-PDF extraction run in a resource-limited subprocess.

Invoked as ``python pdf_extract.py '<json limits>'`` with the PDF bytes on
stdin; writes one JSON object to stdout. It imports only ``pypdf`` and the
standard library (no application settings, database or network client),
never executes PDF JavaScript/actions, and stops at the page limit,
recording the omitted range instead of pretending the document is complete.
"""

from __future__ import annotations

import io
import json
import sys


def extract(data: bytes, max_pages: int) -> dict:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return {"ok": False, "failure": "encrypted_pdf"}
            except Exception:  # noqa: BLE001 - any decrypt failure is terminal
                return {"ok": False, "failure": "encrypted_pdf"}
        total = len(reader.pages)
    except (PdfReadError, ValueError, KeyError, TypeError, OSError):
        return {"ok": False, "failure": "malformed_pdf"}
    try:
        labels = list(reader.page_labels)
    except Exception:  # noqa: BLE001 - labels are optional metadata
        labels = []
    pages = []
    failed_pages = []
    for index in range(min(total, max_pages)):
        try:
            text = reader.pages[index].extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not hide others
            failed_pages.append(index)
            continue
        pages.append(
            {
                "index": index,
                "label": labels[index] if index < len(labels) else str(index + 1),
                "text": text,
            }
        )
    omitted = [] if total <= max_pages else [[max_pages, total - 1]]
    return {
        "ok": True,
        "page_count": total,
        "pages": pages,
        "failed_pages": failed_pages,
        "omitted_ranges": omitted,
        "extractor": "pypdf",
    }


def main() -> None:
    limits = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    data = sys.stdin.buffer.read()
    result = extract(data, int(limits.get("max_pages", 300)))
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
