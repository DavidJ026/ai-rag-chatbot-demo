"""Per-corpus chunking - DESIGN.md §3.3.

Manuals and recipes are chunked differently on purpose. A 2,000-word recipe
embedded as one vector is too diluted to match detail queries, but split naively
its ingredients get orphaned from its method. Recipes are therefore split into
small sections *for embedding only*; the model always receives the whole parent
document (CLAUDE.md rule 8).
"""

import json
import re
from dataclasses import dataclass, field

from .clients import anthropic_client
from .config import (
    CACHE,
    MANUAL_CHUNK,
    MANUAL_CHUNK_MAX,
    MANUAL_OVERLAP,
    MODEL,
    RECIPE_CHUNK,
)

_CALIBRATION = CACHE / "chars_per_token.json"
_SAMPLE = (
    "Preheat the air fryer to 190 C. Toss the chicken thighs with olive oil, "
    "smoked paprika and salt, then cook for 18 minutes, turning once. The "
    "control panel shows E4 when the basket is not fully seated."
) * 6


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chars_per_token():
    """Calibrate once and cache. Never call count_tokens per chunk boundary
    (CLAUDE.md rule 9) - it is an API round-trip and there are thousands."""
    if _CALIBRATION.exists():
        return json.loads(_CALIBRATION.read_text())["chars_per_token"]

    resp = anthropic_client().messages.count_tokens(
        model=MODEL, messages=[{"role": "user", "content": _SAMPLE}]
    )
    ratio = len(_SAMPLE) / resp.input_tokens
    CACHE.mkdir(parents=True, exist_ok=True)
    _CALIBRATION.write_text(json.dumps({"chars_per_token": ratio}, indent=2))
    return ratio


def _paragraphs(text):
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if block:
            yield block


def _split_long(text, max_chars):
    """Break a paragraph that exceeds the window on its own."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    # A single sentence can still be too long (tables, spec lists).
    out = []
    for part in parts:
        while len(part) > max_chars:
            out.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            out.append(part)
    return out


def _pack(units, target_chars, max_chars, overlap_chars):
    """units: [(payload, text)]. Returns a list of unit lists.

    Windows are emitted at target_chars and hard-capped at max_chars. Callers
    must pre-split units to max_chars - overlap_chars (see _split_long) so a
    unit always fits alongside a carried tail.
    """
    windows, current, length = [], [], 0
    for payload, text in units:
        if current and length + len(text) > max_chars:
            windows.append(current)
            current, length = _tail(current, overlap_chars)
        current.append((payload, text))
        length += len(text)
        if length >= target_chars:
            windows.append(current)
            current, length = _tail(current, overlap_chars)
    if current and (not windows or length > overlap_chars):
        windows.append(current)
    return windows


def _tail(window, overlap_chars):
    """The trailing slice of a window, carried into the next one.

    Whole units are preferred, but a unit larger than the overlap budget is
    truncated rather than carried entire - otherwise a long paragraph would be
    duplicated wholesale into the following window.
    """
    if overlap_chars <= 0:
        return [], 0

    tail, length = [], 0
    for payload, text in reversed(window):
        if length + len(text) > overlap_chars:
            break
        tail.insert(0, (payload, text))
        length += len(text)

    if not tail:
        payload, text = window[-1]
        fragment = text[-overlap_chars:]
        return [(payload, fragment)], len(fragment)
    return tail, length


def chunk_manual(doc, pages, cpt):
    """pages: [(page_no, text, has_diagram)] from transcribe.parse_pages."""
    target_chars = int(MANUAL_CHUNK * cpt)
    max_chars = int(MANUAL_CHUNK_MAX * cpt)
    overlap_chars = int(target_chars * MANUAL_OVERLAP)

    units = []
    for page_no, text, has_diagram in pages:
        for paragraph in _paragraphs(text):
            # Leave room for the carried tail so a unit always fits in a window.
            for part in _split_long(paragraph, max_chars - overlap_chars):
                units.append(((page_no, has_diagram), part))

    chunks = []
    windows = _pack(units, target_chars, max_chars, overlap_chars)
    for index, window in enumerate(windows):
        page_numbers = sorted({payload[0] for payload, _ in window})
        chunks.append(
            Chunk(
                id=f"manual::{doc}::{index:04d}",
                text="\n\n".join(text for _, text in window),
                metadata={
                    "source_type": "manual",
                    "doc": doc,
                    "page": page_numbers[0],
                    "pages": ",".join(str(p) for p in page_numbers),
                    "has_diagram": any(payload[1] for payload, _ in window),
                },
            )
        )
    return chunks


def chunk_recipe(doc, text, cpt):
    """Sections exist only to be embedded. The parent document is what the model
    sees, so these windows carry no overlap."""
    target_chars = int(RECIPE_CHUNK * cpt)
    max_chars = int(target_chars * 1.5)
    units = []
    for paragraph in _paragraphs(text):
        for part in _split_long(paragraph, max_chars):
            units.append((None, part))

    chunks = []
    for index, window in enumerate(_pack(units, target_chars, max_chars, 0)):
        chunks.append(
            Chunk(
                id=f"recipe::{doc}::{index:04d}",
                text="\n\n".join(text for _, text in window),
                metadata={"source_type": "recipe", "doc": doc, "parent_id": doc},
            )
        )
    return chunks
