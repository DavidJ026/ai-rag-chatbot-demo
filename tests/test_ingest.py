import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sys, pathlib, tempfile, os, types
for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "APP_PASSWORD"):
    os.environ.setdefault(k, "x")
from harness import Block, Resp, FakeAnthropic, FakeVoyage, make_pdf

ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  <- {detail}" if not cond and detail else ""))

tmp = pathlib.Path(tempfile.mkdtemp())
docs, corpus, chroma, pages, cache = (tmp / n for n in
    ("docs", "corpus", "chroma_db", "pages", "cache"))
(docs / "manuals").mkdir(parents=True)
(docs / "recipes").mkdir(parents=True)
cache.mkdir()

# 2 manuals x 12 pages -> 2 slices each at SLICE_PAGES=10
for name in ("air-fryer", "oven"):
    (docs / "manuals" / f"{name}.pdf").write_bytes(make_pdf(12))
for i in range(3):
    (docs / "recipes" / f"recipe-{i}.txt").write_text(
        "\n\n".join(f"Step {s} for recipe {i}. " + "word " * 80 for s in range(6)))

import ingest
from rag import transcribe, chunking, embeddings as emb

def responder(kw):
    text = kw["messages"][0]["content"][1]["text"]
    first = int(text.split("pages ")[1].split(" to ")[0])
    count = int(text.split("Transcribe all ")[1].split(" of them")[0])
    out = []
    for p in range(first, first + count):
        out.append(f"<!-- page: {p} -->")
        if p % 5 == 0:
            out.append("<!-- diagram -->")
        out.append(f"Content for page {p}. " + "detail " * 90)
        out.append("")
        out.append(f"Second paragraph on page {p}. " + "more " * 90)
        out.append("")
    return Resp([Block("\n".join(out))])

fake = FakeAnthropic(responder=responder)
fake.messages.count_tokens = lambda **kw: types.SimpleNamespace(
    input_tokens=max(1, len(kw["messages"][0]["content"]) // 4))
transcribe.anthropic_client = lambda: fake
chunking.anthropic_client = lambda: fake
emb.voyage_client = lambda: FakeVoyage()
transcribe.CACHE = cache
chunking._CALIBRATION = cache / "cpt.json"
ingest.DOCS, ingest.CORPUS, ingest.CHROMA, ingest.PAGES = docs, corpus, chroma, pages

ingest.build_manuals()
ingest.build_recipes()
ingest.build_index()

check("corpus markdown written for each manual",
      sorted(p.stem for p in corpus.glob("manuals/*.md")) == ["air-fryer", "oven"])
check("recipes copied into corpus (app reads parents from here)",
      len(list(corpus.glob("recipes/*.md"))) == 3)

md = (corpus / "manuals" / "air-fryer.md").read_text()
parsed = transcribe.parse_pages(md)
check("all 12 pages survive slicing and reassembly",
      [p[0] for p in parsed] == list(range(1, 13)), str([p[0] for p in parsed]))

rendered = sorted(p.name for p in (pages / "air-fryer").glob("*.jpg"))
check("only diagram pages rendered", rendered == ["005.jpg", "010.jpg"], str(rendered))
sizes = [p.stat().st_size for p in (pages / "air-fryer").glob("*.jpg")]
check("rendered pages are real JPEGs", all(s > 500 for s in sizes), str(sizes))

import chromadb
coll = chromadb.PersistentClient(path=str(chroma)).get_collection("corpus")
n = coll.count()
check("index populated", n > 0, f"{n} chunks")
got = coll.get(include=["metadatas"])
kinds = {m["source_type"] for m in got["metadatas"]}
check("both corpora indexed", kinds == {"manual", "recipe"}, str(kinds))
check("recipe chunks carry parent_id",
      all("parent_id" in m for m in got["metadatas"] if m["source_type"] == "recipe"))
check("manual chunks carry page + has_diagram",
      all({"page", "has_diagram"} <= set(m) for m in got["metadatas"] if m["source_type"] == "manual"))

# rerun is resumable: no new transcription calls
before = len(fake.calls)
ingest.build_manuals()
check("rerun serves transcription from cache", len(fake.calls) == before,
      f"{len(fake.calls) - before} new calls")

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
