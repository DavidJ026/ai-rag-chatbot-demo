"""Cached client singletons.

Streamlit reruns the whole script on every interaction, so these must be cached
or the app opens new clients per keystroke-submit (CLAUDE.md rule 6).
"""

import functools

from .config import CHROMA, COLLECTION, secret

try:
    import streamlit as st

    cache = st.cache_resource
except ModuleNotFoundError:  # ingest.py runs outside Streamlit
    cache = functools.cache


@cache
def anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"))


@cache
def voyage_client():
    import voyageai

    return voyageai.Client(api_key=secret("VOYAGE_API_KEY"))


@cache
def collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA))
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
