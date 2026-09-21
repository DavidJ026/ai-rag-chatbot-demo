"""Voyage wrapper.

voyage-3.5 is asymmetric: documents and queries must be embedded with different
input_type values or retrieval degrades (CLAUDE.md rule 5).
"""

from .clients import voyage_client
from .config import EMBED_MODEL

BATCH = 128


def embed_documents(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        result = voyage_client().embed(batch, model=EMBED_MODEL, input_type="document")
        out.extend(result.embeddings)
    return out


def embed_query(text):
    result = voyage_client().embed([text], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]
