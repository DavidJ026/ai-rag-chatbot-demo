import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import os, tempfile, types
for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "APP_PASSWORD"):
    os.environ.setdefault(k, "x")
from harness import Block, Cite, Resp, FakeAnthropic

ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  <- {detail}" if not cond and detail else ""))

# --- controlled embeddings: same topic -> distance 0, different -> distance 1
TOPICS = ["airfry", "oven", "beef", "other"]
def topic_of(text):
    low = text.lower()
    for t in TOPICS[:-1]:
        if t in low or (t == "airfry" and "air fry" in low):
            return t
    return "other"

class TopicVoyage:
    def embed(self, texts, model=None, input_type=None):
        assert input_type in ("document", "query"), f"bad input_type {input_type!r}"
        SEEN.add(input_type)
        vecs = []
        for t in texts:
            v = [0.0] * len(TOPICS)
            v[TOPICS.index(topic_of(t))] = 1.0
            vecs.append(v)
        return types.SimpleNamespace(embeddings=vecs)
SEEN = set()

import chromadb
from rag import store, embeddings as emb, router, answer
from rag.config import DISTANCE_FLOOR

tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "recipes").mkdir()
store.CORPUS = tmp
store.PAGES = tmp / "pages"
(tmp / "pages" / "oven-manual").mkdir(parents=True)
(tmp / "pages" / "oven-manual" / "012.jpg").write_bytes(b"fake-jpeg")

client = chromadb.PersistentClient(path=str(tmp / "db"))
coll = client.get_or_create_collection("testcorpus", metadata={"hnsw:space": "cosine"})
emb.voyage_client = lambda: TopicVoyage()
store.collection = lambda: coll

rows = [
    ("m1", "Air fry setting: press Air Fry, set 190C for 18 minutes.",
     {"source_type": "manual", "doc": "oven-manual", "page": 12, "pages": "12", "has_diagram": True}),
    ("m2", "Oven self-clean cycle runs for three hours.",
     {"source_type": "manual", "doc": "oven-manual", "page": 40, "pages": "40", "has_diagram": False}),
]
for i in range(4):
    name = f"beef-recipe-{i}"
    (tmp / "recipes" / f"{name}.md").write_text(f"# Beef recipe {i}\n\nBraised beef, whole document body {i}.")
    for s in range(2):
        rows.append((f"r{i}{s}", f"Beef section {s} of recipe {i}.",
                     {"source_type": "recipe", "doc": name, "parent_id": name}))

coll.add(ids=[r[0] for r in rows],
         embeddings=emb.embed_documents([r[1] for r in rows]),
         documents=[r[1] for r in rows],
         metadatas=[r[2] for r in rows])

# --- one fake Anthropic, dispatching on call shape
state = {"cite": True, "classify": None}
def responder(kw):
    if kw.get("max_tokens") == 5:
        return Resp([Block(state["classify"] or "BOTH")])
    if state["cite"]:
        return Resp([Block("Grounded answer.", [Cite(0, "oven-manual - page 12", "set 190C")])])
    return Resp([Block("Ungrounded answer with no citations.")])

fake = FakeAnthropic(responder=responder, streamer=lambda kw: ["fallback ", "text"])
router.anthropic_client = lambda: fake
answer.anthropic_client = lambda: fake

def run(q, classify, cite=True, history=None):
    state["classify"], state["cite"] = classify, cite
    fake.calls.clear()
    return router.route(q, history or [])

# --- path 1: out of scope
r = run("who won the 1998 world cup", "OUT_OF_SCOPE")
check("out of scope -> refusal", r.stage == "refusal", r.stage)
check("refusal makes no answer call", len(fake.calls) == 1, f"{len(fake.calls)} calls")
check("refusal text is the canned one", r.text == router.REFUSAL)

# --- path 2: grounded
r = run("how do I air fry chicken", "MANUAL", cite=True)
check("grounded -> documents badge", r.stage == "documents", r.stage)
check("citations resolved to units", len(r.citations) == 1 and r.citations[0].unit.doc == "oven-manual")
check("manual unit carries its page image",
      r.citations[0].unit.image_path and r.citations[0].unit.image_path.endswith("012.jpg"),
      str(r.citations[0].unit.image_path))
doc_blocks = [b for b in fake.calls[-1]["messages"][-1]["content"] if b["type"] == "document"]
check("units passed as document blocks", len(doc_blocks) >= 1)
check("citations enabled on every document block",
      all(b["citations"] == {"enabled": True} for b in doc_blocks))

# --- path 3: retrieved but ungrounded -> THE critical fallback trigger
r = run("how do I air fry chicken", "MANUAL", cite=False)
check("retrieved-but-uncited -> general badge", r.stage == "general", r.stage)
check("stage 1 was actually attempted first", len(fake.calls) == 2, f"{len(fake.calls)} calls")
check("fallback streams", "".join(r.stream) == "fallback text")

# --- path 4: nothing clears the floor
r = run("explain quantum chromodynamics", "BOTH")
check("no hits -> general badge", r.stage == "general", r.stage)
check("no hits skips the stage 1 call", len(fake.calls) == 1, f"{len(fake.calls)} calls")

# --- manual disclaimer in the fallback system prompt
# Text that matches no indexed topic, so the fallback is guaranteed.
state["classify"] = "MANUAL"; state["cite"] = True; fake.calls.clear()
list(router.route("what does error code E4 mean", []).stream)
sys_prompt = [c for c in fake.calls if c.get("max_tokens") != 5][-1]["system"]
check("manual fallback carries the stronger disclaimer", "Not from your manual" in sys_prompt)
state["classify"] = "RECIPE"; fake.calls.clear()
list(router.route("how long should I simmer it", []).stream)
sys_prompt = [c for c in fake.calls if c.get("max_tokens") != 5][-1]["system"]
check("recipe fallback omits the manual disclaimer", "Not from your manual" not in sys_prompt)

# --- recipe parent expansion
units = store.retrieve(emb.embed_query("beef stew"), "RECIPE")
check("parents deduped and capped at 3", len(units) == 3, f"got {len(units)}")
check("parents are distinct", len({u.doc for u in units}) == 3)
check("whole parent document is loaded",
      all(u.text.startswith("# Beef recipe") for u in units), units[0].text[:40])

# --- floor + input_type
check("floor filters unrelated topics", store.retrieve(emb.embed_query("quantum"), "BOTH") == [])
check("query and document input_types both exercised", SEEN == {"document", "query"}, str(SEEN))

# --- classifier robustness
state["classify"] = "  recipe.\n"
check("classifier tolerates whitespace/punctuation", router.classify("x", []) == "RECIPE")
state["classify"] = "I think this is about cooking"
check("unparseable classification falls back to BOTH", router.classify("x", []) == "BOTH")

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
