#!/usr/bin/env python3
"""Offline build: docs/ -> corpus/, chroma_db/, pages/

Runs on a developer laptop, never on the server (CLAUDE.md "Ingest is offline").
Resumable: completed transcription slices are cached in .cache/ by content hash,
so a rerun after a failure does not re-pay for finished work.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import chromadb
import pypdfium2 as pdfium

from rag import transcribe
from rag.chunking import chars_per_token, chunk_manual, chunk_recipe
from rag.config import (
    CHROMA,
    COLLECTION,
    CORPUS,
    DOCS,
    INGEST_CONCURRENCY,
    PAGE_DPI,
    PAGE_QUALITY,
    PAGES,
)
from rag.embeddings import embed_documents

ADD_BATCH = 500
STAGES = ("manuals", "recipes", "index")


def log(message):
    print(message, flush=True)


def build_manuals():
    sources = sorted(DOCS.glob("manuals/*.pdf"))
    if not sources:
        log("no PDFs in docs/manuals/ - skipping manuals")
        return

    (CORPUS / "manuals").mkdir(parents=True, exist_ok=True)
    for pdf_path in sources:
        doc = pdf_path.stem
        log(f"\n=== {doc} ===")
        raw_pages = transcribe.extract_text(pdf_path)
        slices = list(transcribe.slice_pdf(pdf_path))
        log(f"  {len(raw_pages)} pages, {len(slices)} slices")

        def run(slice_):
            data, first_page, count = slice_
            hint_pages = raw_pages[first_page - 1 : first_page - 1 + count]
            markdown, missing = transcribe.transcribe_slice(
                data, first_page, count, hint_pages
            )
            if missing:
                log(f"  ! pages still missing after retries: {missing}")
            return first_page, markdown

        with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as pool:
            results = sorted(pool.map(run, slices))

        markdown = "\n\n".join(text for _, text in results)
        (CORPUS / "manuals" / f"{doc}.md").write_text(markdown)

        pages = transcribe.parse_pages(markdown)
        diagram_pages = [page for page, _, has_diagram in pages if has_diagram]
        log(f"  transcribed {len(pages)} pages, {len(diagram_pages)} with diagrams")
        render_pages(pdf_path, diagram_pages, PAGES / doc)


def render_pages(pdf_path, page_numbers, out_dir):
    if not page_numbers:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    for number in page_numbers:
        target = out_dir / f"{number:03d}.jpg"
        if target.exists():
            continue
        bitmap = pdf[number - 1].render(scale=PAGE_DPI / 72)
        bitmap.to_pil().convert("RGB").save(
            target, "JPEG", quality=PAGE_QUALITY, optimize=True
        )


def build_recipes():
    sources = sorted(
        path
        for pattern in ("recipes/*.txt", "recipes/*.md")
        for path in DOCS.glob(pattern)
    )
    if not sources:
        log("no files in docs/recipes/ - skipping recipes")
        return

    out_dir = CORPUS / "recipes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sources:
        # Copied into corpus/ because the deployed app reads whole parent
        # documents from here; docs/ is gitignored.
        (out_dir / f"{path.stem}.md").write_text(path.read_text())
    log(f"copied {len(sources)} recipes into corpus/recipes/")


def build_index():
    cpt = chars_per_token()
    chunks = []

    for path in sorted(CORPUS.glob("manuals/*.md")):
        pages = transcribe.parse_pages(path.read_text())
        chunks += chunk_manual(path.stem, pages, cpt)

    for path in sorted(CORPUS.glob("recipes/*.md")):
        chunks += chunk_recipe(path.stem, path.read_text(), cpt)

    if not chunks:
        log("nothing to index - run the manual and recipe stages first")
        return

    manuals = sum(1 for c in chunks if c.metadata["source_type"] == "manual")
    log(f"\nchunks: {manuals} manual, {len(chunks) - manuals} recipe")

    log("embedding...")
    vectors = embed_documents([chunk.text for chunk in chunks])

    client = chromadb.PersistentClient(path=str(CHROMA))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    store = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), ADD_BATCH):
        window = chunks[start : start + ADD_BATCH]
        store.add(
            ids=[c.id for c in window],
            embeddings=vectors[start : start + ADD_BATCH],
            documents=[c.text for c in window],
            metadatas=[c.metadata for c in window],
        )
    log(f"indexed {len(chunks)} chunks into {CHROMA}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=STAGES,
        help="run only these stages",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=STAGES,
        help="run every stage except these (--skip manuals avoids the "
        "expensive transcription pass but still rebuilds the index)",
    )
    args = parser.parse_args()

    stages = set(args.only or STAGES) - set(args.skip or [])
    if not stages:
        parser.error("nothing left to run")

    # Ordered, because index consumes what the other two produce.
    if "manuals" in stages:
        build_manuals()
    if "recipes" in stages:
        build_recipes()
    if "index" in stages:
        build_index()

    log("\ndone. Commit corpus/, chroma_db/ and pages/ - they are build artifacts.")


if __name__ == "__main__":
    sys.exit(main())
