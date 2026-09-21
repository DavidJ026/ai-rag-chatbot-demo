"""Paths, tunables and secret resolution. See DESIGN.md."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CORPUS = ROOT / "corpus"
CHROMA = ROOT / "chroma_db"
PAGES = ROOT / "pages"
CACHE = ROOT / ".cache"

# DESIGN.md §5 - one model for every call, by decision. Do not add a second.
MODEL = "claude-haiku-4-5"
EMBED_MODEL = "voyage-3.5"
COLLECTION = "corpus"

# Retrieval - DESIGN.md §4.2
MANUAL_TOP_K = 5
RECIPE_TOP_K = 8
RECIPE_MAX_PARENTS = 3

# Chroma cosine distance (1 - cosine similarity). Chunks above this are dropped,
# which routes the question to stage 2. Tune against real questions: too low
# starves stage 1, too high feeds the model chunks that don't contain the answer.
DISTANCE_FLOOR = 0.60

# Chunking, in tokens - DESIGN.md §3.3
MANUAL_CHUNK = 650
MANUAL_CHUNK_MAX = 800
MANUAL_OVERLAP = 0.15
RECIPE_CHUNK = 400

# Ingest - DESIGN.md §3.1
SLICE_PAGES = 10
PAGE_DPI = 100
PAGE_QUALITY = 75
INGEST_CONCURRENCY = 5
TRANSCRIBE_MAX_TOKENS = 8000

# Spend control - DESIGN.md §6
MAX_SESSION_MESSAGES = 20
ANSWER_MAX_TOKENS = 4000


def _toml_secrets():
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            raise RuntimeError(
                "Reading .streamlit/secrets.toml needs Python 3.11+ or `pip "
                "install tomli`. Alternatively export the secrets as "
                "environment variables."
            ) from None

    with path.open("rb") as fh:
        return tomllib.load(fh)


def secret(name):
    """st.secrets -> .streamlit/secrets.toml -> environment.

    The middle hop lets ingest.py run from the same file the app uses, with no
    environment setup.
    """
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    value = _toml_secrets().get(name) or os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing secret {name!r}. Set it in .streamlit/secrets.toml "
            f"(see secrets.toml.example) or export it."
        )
    return value
