import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import os, tempfile
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("VOYAGE_API_KEY", "x")
os.environ.setdefault("APP_PASSWORD", "x")
from harness import *

ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  <- {detail}" if not cond and detail else ""))

# ---------- chunking ----------
from rag import chunking
CPT = 4.0

recipe = "\n\n".join(f"Paragraph {i}. " + ("word " * 60) for i in range(12))
rc = chunking.chunk_recipe("braised-beef", recipe, CPT)
check("recipe splits into multiple sections", len(rc) > 1, f"got {len(rc)}")
check("every recipe chunk carries parent_id",
      all(c.metadata["parent_id"] == "braised-beef" for c in rc))
check("recipe chunk ids unique", len({c.id for c in rc}) == len(rc))
sizes = [len(c.text) for c in rc]
check("recipe sections near 400 tokens",
      all(s <= 400 * CPT * 2.2 for s in sizes), f"sizes {sizes}")

pages = [(1, "Intro. " + "alpha " * 200, False),
         (2, "Controls. " + "beta " * 200, True),
         (3, "Errors. " + "gamma " * 200, False)]
mc = chunking.chunk_manual("oven", pages, CPT)
check("manual splits into multiple chunks", len(mc) > 1, f"got {len(mc)}")
check("manual chunks record first page",
      all(isinstance(c.metadata["page"], int) for c in mc))
check("has_diagram propagates", any(c.metadata["has_diagram"] for c in mc))
check("metadata is chroma-safe (str/int/bool only)",
      all(isinstance(v, (str, int, float, bool))
          for c in mc + rc for v in c.metadata.values()))
joined = " ".join(c.text for c in mc)
check("manual windows overlap", len(joined) > sum(len(p[1]) for p in pages) * 0.99)

MAX = 800 * CPT
huge = "x" * 40000
big = chunking.chunk_manual("big", [(1, huge, False)], CPT)
check("oversized paragraph is split", len(big) > 1, f"got {len(big)}")
check("no manual chunk exceeds MANUAL_CHUNK_MAX",
      all(len(c.text) <= MAX + 64 for c in big + mc),
      f"largest {max(len(c.text) for c in big + mc):.0f} vs cap {MAX:.0f}")
prose = [(1, "\n\n".join("Sentence %d. " % i + "word " * 120 for i in range(30)), False)]
pc = chunking.chunk_manual("prose", prose, CPT)
check("prose chunks respect the cap too",
      all(len(c.text) <= MAX + 64 for c in pc),
      f"largest {max(len(c.text) for c in pc):.0f} vs cap {MAX:.0f}")
total_src = len(prose[0][1])
total_out = sum(len(c.text) for c in pc)
check("overlap duplication stays bounded (<40%)",
      total_out < total_src * 1.4, f"{total_out/total_src:.2f}x")

# ---------- transcribe parsing ----------
from rag import transcribe
md = """<!-- page: 1 -->
Getting started. Plug in the unit.

<!-- page: 2 -->
<!-- diagram -->
Control panel: six buttons left to right: Power, Air Fry, Roast, Bake, Broil, Dehydrate.

<!-- page: 3 -->
(blank)
"""
parsed = transcribe.parse_pages(md)
check("parse_pages skips blank pages", [p[0] for p in parsed] == [1, 2], str(parsed))
check("parse_pages flags diagram page", parsed[1][2] is True)
check("diagram marker stripped from text", "<!-- diagram -->" not in parsed[1][1])
check("validate accepts an explicitly blank page", transcribe.validate(md, 1, 3) == [])
check("validate finds omitted pages", transcribe.validate(md, 1, 5) == [4, 5])
check("validate passes a complete slice", transcribe.validate(md, 1, 2) == [])

# ---------- transcribe retry ----------
from rag.config import CACHE
tmp = tempfile.mkdtemp()
transcribe.CACHE = pathlib.Path(tmp)
attempts = {"n": 0}
def responder(kw):
    attempts["n"] += 1
    if attempts["n"] == 1:
        return Resp([Block("<!-- page: 1 -->\nonly the first page\n")])
    return Resp([Block("<!-- page: 1 -->\nfirst\n\n<!-- page: 2 -->\nsecond\n")])
fake = FakeAnthropic(responder=responder)
transcribe.anthropic_client = lambda: fake
pdf = make_pdf(2)
out, missing = transcribe.transcribe_slice(pdf, 1, 2, ["", ""])
check("retries when a page is missing", attempts["n"] == 2, f"attempts={attempts['n']}")
check("retry recovers the missing page", missing == [], str(missing))
check("retry prompt names the missing page",
      "2" in fake.calls[1]["messages"][0]["content"][1]["text"])
attempts["n"] = 0
out2, _ = transcribe.transcribe_slice(pdf, 1, 2, ["", ""])
check("second call is served from cache", attempts["n"] == 0)
check("cached text matches", out2 == out)

# ---------- slicing ----------
pdf_path = pathlib.Path(tmp) / "big.pdf"
pdf_path.write_bytes(make_pdf(25))
slices = list(transcribe.slice_pdf(pdf_path, pages_per_slice=10))
check("slices cover every page", [(s[1], s[2]) for s in slices] == [(1, 10), (11, 10), (21, 5)],
      str([(s[1], s[2]) for s in slices]))

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
