"""Chroma access and parent expansion - DESIGN.md §4.2."""

from __future__ import annotations

from dataclasses import dataclass

from .clients import collection
from .config import (
    CORPUS,
    DISTANCE_FLOOR,
    MANUAL_TOP_K,
    PAGES,
    RECIPE_MAX_PARENTS,
    RECIPE_TOP_K,
)


@dataclass
class Unit:
    """One thing handed to the model as a document block."""

    title: str
    text: str
    source_type: str
    doc: str
    distance: float
    page: int | None = None
    image_path: str | None = None


def _query(query_embedding, source_type, n_results):
    try:
        result = collection().query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"source_type": source_type},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []
    if not result.get("ids") or not result["ids"][0]:
        return []
    return list(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    )


def _manual_units(query_embedding):
    units = []
    for text, meta, distance in _query(query_embedding, "manual", MANUAL_TOP_K):
        if distance > DISTANCE_FLOOR:
            continue
        page = meta.get("page")
        image = PAGES / meta["doc"] / f"{int(page):03d}.jpg" if page else None
        units.append(
            Unit(
                title=f"{meta['doc']} - page {page}",
                text=text,
                source_type="manual",
                doc=meta["doc"],
                distance=distance,
                page=int(page) if page is not None else None,
                image_path=str(image) if image and image.exists() else None,
            )
        )
    return units


def _recipe_units(query_embedding):
    """Sections are the retrieval unit; whole recipes are the context unit."""
    seen, units = set(), []
    for _text, meta, distance in _query(query_embedding, "recipe", RECIPE_TOP_K):
        if distance > DISTANCE_FLOOR:
            continue
        parent = meta["parent_id"]
        if parent in seen:
            continue
        seen.add(parent)

        path = CORPUS / "recipes" / f"{parent}.md"
        if not path.exists():
            continue
        units.append(
            Unit(
                title=parent,
                text=path.read_text(),
                source_type="recipe",
                doc=parent,
                distance=distance,
            )
        )
        if len(units) >= RECIPE_MAX_PARENTS:
            break
    return units


def retrieve(query_embedding, kind):
    units = []
    if kind in ("MANUAL", "BOTH"):
        units += _manual_units(query_embedding)
    if kind in ("RECIPE", "BOTH"):
        units += _recipe_units(query_embedding)
    return sorted(units, key=lambda u: u.distance)
