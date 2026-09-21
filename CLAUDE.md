# CLAUDE.md

Instructions for AI coding tools working in this repo.

## Project

Password-gated Streamlit RAG chatbot over a personal corpus: 10 appliance manuals
(~100pp each, diagram-heavy PDFs) and 100 recipe documents (~2,000 words each).
Python + Streamlit + ChromaDB + Voyage embeddings + Claude. Deployed to Streamlit
Community Cloud at a public URL behind a shared password.

The bot is **topic-scoped** (appliances and cooking only), answers **from the corpus first**,
falls back to general knowledge when the corpus can't answer, and **badges every answer with
which source it used**.

## Documents

`SPEC.md` is what the product does; `DESIGN.md` is how it is built. Read both before writing
code.

## Hard rules

These cause HTTP 400s or silent quality loss. Do not "improve" them without being asked.

1. **Model is `claude-haiku-4-5` for every call.** Topic gate, RAG answer, fallback answer,
   ingest transcription. This is an explicit owner decision (cheapest model only), not an
   oversight. Do not upgrade it, do not add a "smarter model for hard queries" fallback.
2. **Never pass `output_config.effort`.** Not supported on Haiku 4.5 — returns 400.
3. **Never pass `thinking: {"type": "adaptive"}`.** Haiku 4.5 predates adaptive thinking. It
   takes the older `{"type": "enabled", "budget_tokens": N}` form. This project needs no
   thinking at all — omit the parameter.
4. **Never write a date-suffixed model ID.** `claude-haiku-4-5` is complete as-is.
5. **`input_type` must be set on every Voyage call** — `"document"` at ingest,
   `"query"` at query time. voyage-3.5 is asymmetric; using the same value for both silently
   degrades retrieval.
6. **Wrap every client in `@st.cache_resource`** — Chroma, Anthropic, Voyage. Streamlit
   reruns the whole script on every interaction. Without this you open new clients per
   keystroke-submit.
7. **The auth gate is the first executable thing in `app.py`.** `hmac.compare_digest`,
   then `st.stop()` if unauthenticated, before any widget that can render corpus content.
8. **Never split a recipe across context units.** Recipes are chunked into ~400-token
   sections *for embedding only*; the model always receives whole parent documents. See
   `DESIGN.md` §3.3 and §4.2.
9. **Never call `count_tokens` per chunk boundary.** Calibrate a chars-per-token ratio once,
   then split on characters.
10. **Never use `tiktoken`** to count tokens for Claude. It is the wrong tokenizer.

## Answer routing — invariants

Three stages, in `rag/router.py`. Full detail in `DESIGN.md` §4. These are the parts that
break quietly if reimplemented casually:

11. **Out-of-scope questions never reach an answer call.** The stage-0 gate returns a single
    token (`RECIPE` / `MANUAL` / `BOTH` / `OUT_OF_SCOPE`); `OUT_OF_SCOPE` short-circuits to a
    canned refusal. This is both a scoping rule and a spend control. Do not replace it with a
    system-prompt instruction on the answer call.
12. **RAG is always tried first.** Never call the fallback without a corpus attempt.
13. **Fallback triggers on two conditions**, not one: nothing cleared the similarity floor,
    **or** the RAG response came back with zero citation-bearing text blocks. The second
    matters — chunks can clear the floor and still not contain the answer. Detect it from the
    citations, not by string-matching the response for "I don't know".
14. **Never blend sources in one answer.** An answer is wholly stage 1 or wholly stage 2. If
    the corpus answers partially, stay in stage 1, badge it as documents, and say what was not
    found. Blending makes the badge meaningless.
15. **Every answer carries a provenance badge**, above the text, not as a footnote:
    🟢 *From your documents* / ⚪ *From general knowledge — not in your documents* /
    ⚪ refusal. This is a core requirement, not UI polish.
16. **Manual fallbacks get the stronger disclaimer** (`DESIGN.md` §4.3) — outside knowledge
    about an appliance is about *some* model, not the user's, and wrong temperatures or
    settings have real-world consequences.
17. **The badge says "general knowledge", not "internet".** There is no web search tool in
    this design. Do not add one without being asked, and do not label trained knowledge as
    internet sources.
18. **Stage 1 is not streamed; stage 2 is.** Whether the stage 1 answer is shown depends on
    the finished response carrying citations, so it cannot be displayed as it arrives.
    Do not "fix" this by streaming it - that shows the user text the system then retracts.

## Ingest is offline

`ingest.py` runs on a developer laptop, never on the server. `app.py` is strictly read-only
against the committed `chroma_db/`. Do not add index-building, embedding of documents, or
PDF processing to the request path.

Ingest must stay **resumable**: cache completed transcription slices to `.cache/` keyed by
content hash. Re-running after a failure must not re-pay for completed work.

Validate transcription output — assert every `<!-- page: N -->` marker is present and each
page's markdown is non-trivial, and retry failed slices. Haiku's dominant failure mode on
long vision transcription is silently skipping pages, and every page lost to that becomes a
question that falls through to the weakest stage of the system.

## Committed vs gitignored

Committed build artifacts are **deliberate**, not accidental:

| Committed | Gitignored |
|---|---|
| `corpus/` transcribed markdown | `docs/` source PDFs and recipe files |
| `chroma_db/` prebuilt index (~6 MB) | `.cache/` transcription cache |
| `pages/` diagram-page JPEGs only (~20 MB) | `.streamlit/secrets.toml` |

Do not add `chroma_db/` or `pages/` to `.gitignore` — Streamlit Community Cloud rebuilds its
filesystem on every redeploy and on wake-from-sleep, so a committed index is the only thing
that survives.

Do not commit source PDFs or non-diagram page renders. They are 50–150 MB and would make
every deploy clone slow.

## Secrets

`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `APP_PASSWORD` — read via `st.secrets`. Local values
live in `.streamlit/secrets.toml` (gitignored) and are mirrored into the Streamlit Cloud
secrets UI. Never hardcode a key, never log one, never write one into a committed file.

## Cost awareness

The app sits on a public URL behind a password that will get forwarded. A RAG query is
~$0.012; an out-of-scope query is ~$0.0003. Keep the per-session message cap in `app.py`
(~20 messages) — it is a spend control, not a UX flourish. Do not remove it.

Any change that increases per-query token count (larger top-k, more parent documents,
passing page images by default, adding web search) is a cost change. Flag it rather than
landing it silently.

## Quality invariants

- **The similarity floor routes, it does not refuse.** Below the floor means stage 2, not
  "not found".
- **Citations stay on.** Retrieved units are passed as `document` content blocks with
  `citations: {"enabled": true}`, not pasted into a text block. Citations are load-bearing
  twice: they show the user what is grounded, and their absence is the fallback trigger
  (rule 13). Do not swap them for prompt-engineered `[1]` markers.
- **System prompt rules are load-bearing** (`DESIGN.md` §4.5): stage 1 answers only from
  context, presents multiple matching recipes separately and never merges them, and
  reproduces quantities, temperatures and times verbatim; stage 2 states up front that it is
  not from the user's documents and invents no specifics about their appliance model.

## Commands

```bash
pip install -r requirements.txt

python ingest.py                  # full offline build (resumable)
python ingest.py --skip manuals   # recipes + index, no transcription pass
python ingest.py --only index     # re-chunk and re-embed what is already transcribed
streamlit run app.py              # local dev; needs .streamlit/secrets.toml
```

## Tests

```bash
python tests/run.py
```

No network, no API keys, no spend - the Anthropic and Voyage clients are stubbed and Chroma
runs in a temp directory. The suite covers chunk-size ceilings, transcription validation and
retry, all four routing paths, a full ingest, and the app through Streamlit's `AppTest`
(auth gate, badges, session cap). Add to it rather than around it.

## Definition of done

`python tests/run.py` passes.

A change to retrieval or chunking is not done until `ingest.py` has been re-run and the
resulting `chroma_db/` committed — the index is a build artifact, and stale indexes fail
silently rather than loudly.

A change to routing is not done until all four badge paths still pass in
`tests/test_routing.py`, and have been spot-checked by hand against the real corpus: an
in-corpus recipe question, an in-corpus manual question, an in-scope question the corpus
cannot answer, and an off-topic question.
