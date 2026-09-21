"""Vision transcription of manual PDFs - DESIGN.md §3.1.

Slices are short because Haiku compresses and drops content over long outputs.
Every slice is validated for missing page markers and retried, because silently
skipping pages is the dominant failure mode and a lost page becomes a question
that falls through to the weakest stage of the system.
"""

import base64
import hashlib
import io
import re

from pypdf import PdfReader, PdfWriter

from .clients import anthropic_client
from .config import CACHE, MODEL, SLICE_PAGES, TRANSCRIBE_MAX_TOKENS

PAGE_MARKER = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
DIAGRAM_MARKER = re.compile(r"<!--\s*diagram\s*-->")

SYSTEM = """You transcribe pages from appliance manuals into markdown.

For EACH page, in order, output a marker line:
<!-- page: N -->
followed by that page's content as markdown.

Rules:
- Use the exact page numbers you are given. Every page marker must appear.
- Transcribe all text faithfully. Keep tables as markdown tables. Never summarise.
- Describe every diagram, photo and icon in prose, in enough detail that someone
  could answer questions about it without seeing it. Name buttons, controls,
  settings and parts exactly as they are labelled.
- If a page is mostly visual - a control panel layout, an exploded parts view, a
  wiring or assembly diagram - put <!-- diagram --> on the line right after its
  page marker.
- If a page is blank, output its marker followed by: (blank)

Output only the markers and the transcribed content."""


def slice_pdf(path, pages_per_slice=SLICE_PAGES):
    """Yield (pdf_bytes, first_page, page_count). Page numbers are 1-based."""
    reader = PdfReader(str(path))
    total = len(reader.pages)
    for start in range(0, total, pages_per_slice):
        writer = PdfWriter()
        for index in range(start, min(start + pages_per_slice, total)):
            writer.add_page(reader.pages[index])
        buffer = io.BytesIO()
        writer.write(buffer)
        yield buffer.getvalue(), start + 1, min(pages_per_slice, total - start)


def extract_text(path):
    """Raw per-page text. Empty for scanned pages; that is expected."""
    reader = PdfReader(str(path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _hint(first_page, count, raw_pages):
    last = first_page + count - 1
    lines = [
        f"This PDF contains pages {first_page} to {last} of the manual. "
        f"Transcribe all {count} of them, in order.",
        "",
        "Text extracted from the PDF layer follows. It is a hint for exact "
        "wording and may be empty for scanned pages - trust the page images for "
        "layout, diagrams and anything the extraction missed.",
        "",
    ]
    for offset, text in enumerate(raw_pages):
        lines.append(f"--- extracted page {first_page + offset} ---")
        lines.append(text if text else "(no extractable text)")
    return "\n".join(lines)


def validate(markdown, first_page, count):
    """Return the page numbers that are missing or effectively empty."""
    found = {}
    parts = PAGE_MARKER.split(markdown)
    for index in range(1, len(parts), 2):
        found[int(parts[index])] = parts[index + 1].strip()

    missing = []
    for page in range(first_page, first_page + count):
        body = DIAGRAM_MARKER.sub("", found.get(page, "")).strip()
        if len(body) < 3:
            missing.append(page)
    return missing


def transcribe_slice(pdf_bytes, first_page, count, raw_pages, attempts=3):
    cache_key = hashlib.sha256(pdf_bytes).hexdigest()[:32]
    cache_path = CACHE / "slices" / f"{cache_key}.md"
    if cache_path.exists():
        return cache_path.read_text(), []

    payload = base64.standard_b64encode(pdf_bytes).decode()
    missing = []
    for attempt in range(attempts):
        hint = _hint(first_page, count, raw_pages)
        if missing:
            hint += (
                f"\n\nYour previous attempt omitted pages "
                f"{', '.join(str(p) for p in missing)}. Include every page."
            )
        response = anthropic_client().messages.create(
            model=MODEL,
            max_tokens=TRANSCRIBE_MAX_TOKENS,
            temperature=0,
            system=SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": payload,
                            },
                        },
                        {"type": "text", "text": hint},
                    ],
                }
            ],
        )
        markdown = "".join(b.text for b in response.content if b.type == "text")
        missing = validate(markdown, first_page, count)
        if not missing or attempt == attempts - 1:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(markdown)
            return markdown, missing
    return markdown, missing


def parse_pages(markdown):
    """Return [(page_no, text, has_diagram)] in page order."""
    pages = []
    parts = PAGE_MARKER.split(markdown)
    for index in range(1, len(parts), 2):
        page_no = int(parts[index])
        body = parts[index + 1]
        has_diagram = bool(DIAGRAM_MARKER.search(body))
        text = DIAGRAM_MARKER.sub("", body).strip()
        if text and text != "(blank)":
            pages.append((page_no, text, has_diagram))
    return sorted(pages, key=lambda p: p[0])
