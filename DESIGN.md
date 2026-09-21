# RAG Chatbot — System Design

How the product in `SPEC.md` is built.

---

## 1. Architecture

Two programs. Ingestion is an **offline build step** run on a developer laptop; the deployed
app is **read-only** against the artifact it produces.

```
  OFFLINE (laptop, run manually)               DEPLOYED (Streamlit Community Cloud)
  ───────────────────────────────              ────────────────────────────────────

  docs/manuals/*.pdf  (gitignored)
         │
         ├─ pypdf: split into 10-page slices
         ├─ pypdf: extract raw text
         │
         ▼
   ┌───────────────┐
   │ Claude Haiku  │  page image + raw text
   │  4.5 (vision) │  ──► markdown w/ page markers
   └───────────────┘      + diagram descriptions
         │
         ▼
   corpus/manuals/*.md ──┐
   docs/recipes/*.txt ───┤
                         │
                         ├─ chunk (per-corpus rules, §3.3)
                         ▼
                  ┌──────────────────┐
                  │ voyage-3.5       │
                  │ input_type=      │                 question
                  │   document       │                    │
                  └──────────────────┘                    ▼
                         │                        ┌───────────────┐
                         ▼                        │ ANSWER ROUTER │
                  chroma_db/  ──────────────────► │     (§4)      │
                  corpus/                         └───────────────┘
                  pages/*.jpg (diagrams only)             │
                         │                                ▼
                         └── committed to git    badged answer
```

---

## 2. Repo layout

```
ai-rag-chatbot-demo/
├── ingest.py                  # offline build: docs/ -> corpus/, chroma_db/, pages/
├── app.py                     # Streamlit: auth gate -> chat loop -> badge rendering
├── rag/
│   ├── router.py              # three-stage routing (§4)
│   ├── chunking.py            # per-corpus chunking rules (§3.3)
│   ├── embeddings.py          # Voyage wrapper, input_type-aware
│   ├── store.py               # Chroma open / query / parent expansion
│   ├── transcribe.py          # vision transcription + validation + retry
│   └── answer.py              # Claude calls: document blocks, citations, streaming
├── docs/                      # SOURCE corpus — gitignored
│   ├── manuals/*.pdf
│   └── recipes/*.txt
├── corpus/                    # transcribed markdown — committed
├── chroma_db/                 # prebuilt index — committed
├── pages/                     # diagram page JPEGs only — committed
├── tests/                     # no network, no API spend - python tests/run.py
├── .cache/                    # transcription slice cache — gitignored
├── requirements.txt
└── .streamlit/secrets.toml    # gitignored
```

`corpus/`, `chroma_db/` and `pages/` are committed build artifacts. Streamlit Community
Cloud rebuilds its filesystem on every redeploy and on wake-from-sleep, so a committed index
is the only thing that survives. Committing it also makes deploys deterministic, cold starts
instant, and re-ingestion an explicit reviewable commit.

---

## 3. Ingest pipeline (`ingest.py`)

Offline, manual, resumable. Never runs on the server.

### 3.1 Manuals

1. **Slice** each 100-page PDF into 10-page PDFs with `pypdf`. Haiku compresses and drops
   content over long outputs, so slices stay short. ~100 requests total; run ~5 concurrent,
   or via the Batch API for half price.
2. **Extract** raw text per page with `pypdf` (empty for scanned pages).
3. **Transcribe** each slice with `claude-haiku-4-5`: send the slice as a base64 `document`
   block *plus* the extracted raw text, so the model interprets layout and diagrams rather
   than re-reading characters. This is the largest quality lever in the pipeline.
   Output is markdown with `<!-- page: N -->` markers, prose descriptions of every diagram,
   and a `<!-- diagram -->` flag on pages whose content is primarily visual.
   `temperature=0`, `max_tokens=8000`.
4. **Validate**: assert every expected page marker is present and each page's markdown is
   non-trivial. Retry failed slices. Haiku's dominant failure mode on long vision
   transcription is silently skipping pages; this catches it for pennies.
5. **Cache** each completed slice to `.cache/` keyed by content hash so reruns skip work.
6. **Render** JPEGs (100 DPI, quality 75) with `pypdfium2` for `<!-- diagram -->` pages only.

### 3.2 Recipes

Plain text, no transcription needed.

### 3.3 Chunking

| Corpus | Unit | Rule |
|---|---|---|
| Manuals | section | 500–800 tokens, ~15% overlap, split on markdown headings where possible |
| Recipes | section | ~400 tokens, each tagged `parent_id` = source document |

Recipe **parent documents are stored whole** and are what reaches the model; sections exist
only to be embedded. A 2,000-word recipe embedded as one vector is too diluted to match
detail queries, but split naively the ingredients get orphaned from the method.

Calibrate a chars-per-token ratio once with `messages.count_tokens`, then split on
characters. Never call `count_tokens` per candidate boundary.

### 3.4 Metadata

Every chunk carries: `source_type` (`manual` | `recipe`), `doc`, `parent_id` (recipes),
`page` + `image_path` + `has_diagram` (manuals).

### 3.5 Embedding

`voyage-3.5`, `input_type="document"`, 1024 dims, batches of 128. ~1,400 chunks → ~11 API
calls, seconds, negligible cost.

---

## 4. Answer routing (`rag/router.py`)

Every question passes through three stages. The stage that produces the answer determines
the badge.

```
  question
     │
     ▼
 ┌────────────────────────────────┐
 │ STAGE 0 — topic gate           │   claude-haiku-4-5, max_tokens=5
 │ RECIPE | MANUAL | BOTH |       │   single-token classification
 │ OUT_OF_SCOPE                   │
 └────────────────────────────────┘
     │                    │
     │ OUT_OF_SCOPE       │ in scope  ──► also sets the Chroma `where` filter
     ▼                    ▼
 ┌──────────┐    ┌────────────────────────────────┐
 │ REFUSAL  │    │ STAGE 1 — RAG                  │
 │ ⚪ badge │    │ embed → search → floor →       │
 │ no answer│    │ parent expansion → answer with │
 │ call     │    │ document blocks + citations    │
 └──────────┘    └────────────────────────────────┘
                     │                      │
                     │ has citations        │ nothing cleared the floor
                     │                      │   OR zero citations returned
                     ▼                      ▼
              ┌──────────────┐     ┌────────────────────────────┐
              │ 🟢 FROM YOUR │     │ STAGE 2 — general knowledge│
              │   DOCUMENTS  │     │ no document blocks         │
              │ + citations  │     │ ⚪ FROM GENERAL KNOWLEDGE  │
              │ + page images│     │   (not in your documents)  │
              └──────────────┘     └────────────────────────────┘
```

### 4.1 Stage 0 — topic gate

A single `claude-haiku-4-5` call, `temperature=0`, `max_tokens=5`, returning exactly one
token: `RECIPE`, `MANUAL`, `BOTH`, or `OUT_OF_SCOPE`. A single token needs no feature
support, costs ~$0.0003, and cannot fail to parse.

`OUT_OF_SCOPE` short-circuits to the canned refusal in `SPEC.md` — **no answer call is
made**. This is both the scoping rule and a spend control.

The same token sets the `source_type` filter for the Chroma query.

### 4.2 Stage 1 — RAG

1. Embed the question — `voyage-3.5`, **`input_type="query"`**. voyage-3.5 is asymmetric;
   using `"document"` for queries measurably degrades retrieval.
2. Chroma similarity search, filtered by the stage-0 `source_type`.
3. **Similarity floor.** Drop chunks past a distance threshold.
4. **Parent expansion** (recipes): top-8 sections → dedupe by `parent_id` → cap at 3 → load
   full parent documents. Manuals: top-5 sections used directly.
5. One `document` content block per retrieved unit, each with `citations: {"enabled": true}`.
6. Render citations as footnotes and, for manual citations, the source page JPEG beneath.

**Stage 1 is not streamed.** Whether its answer is shown at all depends on whether the
finished response carries citations (§4.3), so nothing can be displayed until the response
is complete. Streaming it and then retracting an ungrounded answer would show the user text
the system has already decided not to trust. The UI covers the wait with a spinner; stage 2,
which has no downstream decision, streams normally.

### 4.3 Stage 2 — general-knowledge fallback

Entered when **either**:

- nothing cleared the similarity floor, **or**
- the stage-1 response came back with **zero citation-bearing text blocks**.

The second condition matters: chunks can clear the floor and still not contain the answer.
Citation presence is a native signal for that — no extra call, no string matching on
"I don't know".

Stage 2 re-asks the question with **no document blocks** and the fallback system prompt.

**Manual queries get a stronger disclaimer.** Outside knowledge about an appliance is about
*some* model, not the user's — temperatures, button layouts and error codes differ, and a
confident wrong answer about settings has real-world consequences:

> Not from your manual — this is general information about this type of appliance and may
> not match your model.

Recipe fallbacks use the plain badge.

### 4.4 Never blend sources

**An answer is wholly stage 1 or wholly stage 2.** A blended answer makes the badge
meaningless — the reader cannot tell which half is grounded. If the corpus answers only
partially, stay in stage 1, badge it as documents, and state what was not found.

### 4.5 System prompts

**Stage 1 (RAG).** Answer only from the provided documents; say plainly when the answer is
not there. If multiple recipes match, present them **separately** — never merge steps or
ingredients across recipes. Reproduce quantities, temperatures and times **verbatim**; do not
paraphrase numbers.

**Stage 2 (fallback).** Answer from general knowledge. State up front that this is not from
the user's documents. Do not invent specifics about the user's particular appliance model.
Stay within appliances and cooking.

Cross-recipe contamination — blending two chicken recipes into one plausible-looking answer —
is the primary quality risk in stage 1. Defences: whole-recipe context units, a cap of 3,
citations making the source of each step visible, and the prompt rule above.

---

## 5. Model configuration

**`claude-haiku-4-5` for every call** — topic gate, RAG answer, fallback answer, and ingest
transcription.

Haiku 4.5 predates several current API features. These are hard constraints:

| Constraint | Detail |
|---|---|
| **No `output_config.effort`** | Returns HTTP 400 on Haiku 4.5. Omit entirely. |
| **No adaptive thinking** | Haiku 4.5 uses the older `thinking: {"type": "enabled", "budget_tokens": N}`. Not needed here — omit `thinking`; it is off by default. |
| **200K context** | Not 1M. Never binds: queries are ~9K tokens, transcription slices ~30K. |
| **100-page PDF limit** | Applies to 200K-context models. Never binds: slices are 10 pages. |
| **Sampling allowed** | `temperature` works. Used at 0 for the topic gate and transcription. |

| Call | Config |
|---|---|
| Topic gate | `temperature=0`, `max_tokens=5`, no streaming |
| RAG answer | not streamed (§4.2), `max_tokens=4000`, citations on, no `thinking`, no `effort` |
| Fallback answer | streaming, `max_tokens=4000`, no document blocks |
| Transcription | `temperature=0`, `max_tokens=8000`, 10-page slices |

### 5.1 Quality profile

Stage 1 plays to Haiku's strength: grounded Q&A over supplied context, where retrieval does
most of the work.

**Stage 2 is the weakest part of the system.** Haiku's knowledge of specific appliance models
is thin, and this is exactly where it speaks without grounding. The badge and the
manual-specific disclaimer (§4.3) are the mitigation, not decoration.

**Transcription errors are permanent.** A page transcribed wrong is wrong in the index
forever, retrieval cannot recover it, and the resulting gap pushes queries into stage 2 where
they are least reliable. The validation and retry in §3.1 exist for this reason.

---

## 6. Auth and abuse controls

**Password gate.** `hmac.compare_digest` against `st.secrets["APP_PASSWORD"]`, sets
`st.session_state["authed"]`, and calls `st.stop()` if unset. This must be the **first thing
in `app.py`**, before any widget that could render corpus content.

The gate protects the app, not the repo. One password guards everything, so a leaked password
exposes the whole corpus — acceptable because the corpus is non-sensitive.

**Spend controls.**

1. **Spend limit on the Anthropic API key in the Console.** The only control that cannot be
   bypassed by an app bug. Required.
2. **Per-session message cap** in `st.session_state` (~20).
3. The stage-0 topic gate: off-topic traffic costs ~$0.0003 and never reaches an answer call.

---

## 7. Deployment

Streamlit Community Cloud, public GitHub repo, public HTTPS URL.

**Secrets** — `.streamlit/secrets.toml` locally (gitignored), mirrored into the Cloud secrets
UI: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `APP_PASSWORD`.

**Streamlit reruns the entire script on every interaction.** The Chroma client, the Anthropic
client, and the Voyage client **must** be wrapped in `@st.cache_resource`. Without it every
keystroke-submit opens new clients.

**Artifact sizes.** Source PDFs (~50–150 MB) and non-diagram page renders are gitignored.
Committed: `corpus/` markdown, `chroma_db/` (~6 MB), and diagram-page JPEGs only (~30–40% of
1,000 pages at 100 DPI / q75 ≈ 20 MB). Repo lands ~30 MB — fast clones, well inside Community
Cloud's 1 GB app limit.

---

## 8. Cost model

| Item | Cost |
|---|---|
| Transcription, 1,000 pages (~1.5M in / ~400K out) | **~$3.50** one-time — **~$1.75** via Batch API |
| Embedding, ~1,400 chunks | negligible |
| Topic gate, per query | ~$0.0003 |
| Out-of-scope query (gate only) | **~$0.0003** |
| Stage 2 fallback query (~300 in / ~500 out) | **~$0.003** |
| Stage 1 RAG query (~9K in / ~500 out) | **~$0.012** |
| 1,000 RAG queries | ~$12 |

Ingest is offline and not latency-sensitive, so the Batch API's 50% discount is free money.

---

## 9. Not built

Scope boundaries from `SPEC.md`, plus the implementation-level ones:

- **Live web search** — billed per search on top of tokens, adds latency. Stage 2 uses
  trained knowledge only, which is why its badge says "general knowledge" and shows no URLs.
- **Multimodal embeddings** (`voyage-multimodal-3`) — inflates the index, pays image tokens
  on every query, unreliable for detail lookups. Ingest-time transcription (§3.1) captures
  diagram content more cheaply.
- **Page images passed to the model at query time** for `has_diagram` chunks. Images are
  shown to the user, not sent to the model.
- **Evaluation.** Retrieval quality, transcription quality and topic-gate accuracy are
  unmeasured. Given §5.1, a hand-labelled set — manual lookups, recipe lookups, in-scope
  questions the corpus cannot answer, and off-topic questions — is the first thing to add if
  this outgrows POC status.
