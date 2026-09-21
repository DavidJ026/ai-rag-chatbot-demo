# ai-rag-chatbot-demo

A password-gated chatbot that answers questions about my appliance manuals and my recipes.

Ask it something about your appliances or your cooking and it answers from your own
documents, with citations and the relevant manual page. If your documents don't cover it,
it falls back to general knowledge and says so. Anything off-topic gets turned away.

Every answer is badged with where it came from:

| Badge | Meaning |
|---|---|
| 🟢 **From your documents** | Answered from the indexed corpus, with citations |
| ⚪ **From general knowledge** | Not in your documents |
| ⚪ **Refusal** | Off-topic |

**Stack:** Python · Streamlit · ChromaDB · Voyage `voyage-3.5` · `claude-haiku-4-5` ·
deployed on Streamlit Community Cloud.

---

## What's in here

Two entry points. `ingest.py` runs on your laptop and builds the search index; `app.py` is
the deployed web app and only ever reads it.

| File | Lines | What it does |
|---|---|---|
| **[ingest.py](ingest.py)** | 168 | The offline build. Slices each PDF, sends it to Claude for transcription, validates and retries, renders diagram pages to JPEG, chunks everything, embeds it, and writes the Chroma index. Resumable — completed work is cached. |
| **[app.py](app.py)** | 142 | The Streamlit UI. Password gate, chat loop, provenance badges, citation footnotes and manual page images, per-session message cap. |

The `rag/` package holds the logic both entry points share:

| Module | Lines | Responsibility |
|---|---|---|
| **[rag/router.py](rag/router.py)** | 95 | The three stages: topic gate → RAG → general-knowledge fallback. Decides which badge an answer gets. Start here to understand the system. |
| **[rag/answer.py](rag/answer.py)** | 127 | Both Claude answer calls. Builds document blocks with citations enabled, extracts citations from the response, streams the fallback. Holds both system prompts. |
| **[rag/store.py](rag/store.py)** | 103 | Chroma queries, the similarity floor, and recipe parent expansion (retrieve small sections, hand the model whole recipes). |
| **[rag/chunking.py](rag/chunking.py)** | 183 | Per-corpus chunking. Manuals get overlapping 500–800 token windows; recipes get ~400 token sections tagged with their parent document. |
| **[rag/transcribe.py](rag/transcribe.py)** | 150 | PDF slicing, text extraction, the vision transcription call, and the page-marker validation that catches Claude silently skipping pages. |
| **[rag/embeddings.py](rag/embeddings.py)** | 24 | Voyage wrapper. Exists mainly to keep `input_type` correct — documents and queries must be embedded differently. |
| **[rag/clients.py](rag/clients.py)** | 40 | Cached Anthropic / Voyage / Chroma clients. Streamlit reruns the whole script on every keystroke, so these must be cached. |
| **[rag/config.py](rag/config.py)** | 88 | Every path and tunable in one place, plus secret resolution. **This is the file you edit to tune the system.** |

And the tests:

| | Lines | |
|---|---|---|
| **[tests/run.py](tests/run.py)** | 20 | Runs everything: `python tests/run.py` |
| [tests/test_chunking.py](tests/test_chunking.py) | 114 | Chunk size ceilings, overlap bounds, transcription validation and retry |
| [tests/test_routing.py](tests/test_routing.py) | 148 | All four routing paths against a real Chroma index |
| [tests/test_ingest.py](tests/test_ingest.py) | 97 | A full ingest run end to end |
| [tests/test_app.py](tests/test_app.py) | 109 | The real app through Streamlit's `AppTest` — gate, badges, session cap |

The API clients are stubbed throughout, so the suite needs no keys and costs nothing.

---

## Running it locally, first time

### 1. Get the two API keys

- **Anthropic** — [console.anthropic.com](https://console.anthropic.com) → API keys.
  While you are there, **set a spend limit**. The deployed app is public and the password
  gets shared; this is the only control that can't be bypassed by a bug in the app.
- **Voyage** — [dashboard.voyageai.com](https://dashboard.voyageai.com) → API keys.

### 2. Set up Python

**Needs Python 3.10 or newer** — the `anthropic` SDK requires it. macOS ships 3.9, which is
too old, so you need a newer interpreter from one of these:

```bash
uv venv --python 3.12 .venv           # uv downloads the interpreter for you
brew install python@3.12              # then: python3.12 -m venv .venv
                                      # or install 3.12 from python.org
```

uv is not a project dependency — it is just the easiest way to get a Python on macOS. Any
3.10+ interpreter works, and the deployment uses plain pip.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On 3.10 exactly, `requirements.txt` pulls in `tomli` to read the secrets file; 3.11+ has it
built in.

### 3. Add your secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit it:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
VOYAGE_API_KEY = "pa-..."
APP_PASSWORD = "whatever you want to share"
```

This file is gitignored and never leaves your machine. Both `ingest.py` and `app.py` read
from it, so there is nothing to export.

### 4. Check it works before spending anything

```bash
python tests/run.py
```

Stubbed APIs, no network, no spend. If this passes, your install is good.

### 5. Add your documents

```
docs/
├── manuals/     your appliance PDFs
└── recipes/     your recipes, one .txt or .md file each
```

`docs/` is gitignored — the source files stay on your laptop.

### 6. Build the index

Start with **one** manual to check transcription quality before paying for all of them:

```bash
python ingest.py
```

Then open `corpus/manuals/*.md` and read a few pages. This is what the bot will actually
search, so if diagrams came out vague or tables came out mangled, that is the ceiling on
every answer — worth catching now rather than after the full run. Once you are happy, add
the rest of the PDFs and run it again; finished work is cached, so you only pay for the new
files.

Roughly $3.50 and 15 minutes for 1,000 pages.

If you only changed recipes, `python ingest.py --skip manuals` avoids the transcription pass
but still rebuilds the index — which you need, or the app keeps searching the old one.

### 7. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter your `APP_PASSWORD` and ask it something.

### 8. Tune the one number that matters

`DISTANCE_FLOOR` in [rag/config.py](rag/config.py) decides when a question falls through
from your documents to general knowledge. Ask it ten real questions and watch the badges:

- Questions you *know* are covered coming back grey → the floor is too tight, raise it.
- Obviously unrelated answers badged green → too loose, lower it.

Re-tuning it needs no re-ingest.

---

## Hosting it on the internet

Streamlit Community Cloud is free and gives you a public HTTPS URL.

### 1. Commit the build artifacts

This is the step people get wrong. The index is **not** built on the server — Community
Cloud wipes its filesystem on every redeploy and whenever the app wakes from sleep. The
committed index is the only thing that survives:

```bash
git add corpus/ chroma_db/ pages/
git add app.py ingest.py rag/ tests/ requirements.txt .streamlit/secrets.toml.example
git commit -m "RAG chatbot"
```

`docs/`, `.cache/`, `.venv/` and `secrets.toml` are all gitignored and stay behind. Expect
roughly 30 MB committed.

### 2. Push to GitHub

A public repo is fine — nothing sensitive is committed, and your secrets live in Streamlit's
encrypted store, not in the repo.

```bash
git remote add origin git@github.com:<you>/ai-rag-chatbot-demo.git
git push -u origin main
```

### 3. Deploy

Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and create a
new app:

- **Repository:** your repo
- **Branch:** `main`
- **Main file path:** `app.py`
- **Python version:** 3.11 or newer (under *Advanced settings*)

### 4. Add your secrets to the deployment

Still in *Advanced settings*, paste the **contents** of your local
`.streamlit/secrets.toml` into the secrets box — the same three keys, same TOML format.
These are encrypted and are not visible in the repo.

Then deploy. First boot takes a couple of minutes while it installs dependencies.

### 5. Share it

Send people the URL and the password. Bear in mind the password will be forwarded further
than you intend — that is why step 1 of local setup is the spend limit.

### Updating it later

Changing app code is just `git push`; Community Cloud redeploys automatically.

**Changing documents, chunking or retrieval means re-running `ingest.py` and committing the
regenerated `chroma_db/`.** A stale index fails silently — the app keeps answering, just
from the old corpus.

---

## Costs

| | |
|---|---|
| Building the index, 1,000 pages | ~$3.50 once (~$1.75 via the Batch API) |
| A question answered from your documents | ~$0.012 |
| A question answered from general knowledge | ~$0.003 |
| An off-topic question | ~$0.0003 — it never reaches an answer call |

---

## Docs

| File | Contents |
|---|---|
| [SPEC.md](SPEC.md) | What the product does — behaviour, scope, constraints |
| [DESIGN.md](DESIGN.md) | How it is built — architecture, ingest, routing, costs |
| [CLAUDE.md](CLAUDE.md) | Rules for AI coding tools working in this repo |

## Status

Proof of concept. Retrieval quality is unmeasured, there is no per-user access control and
no hybrid search. The general-knowledge fallback is the weakest part of the system — it is
badged and disclaimed for exactly that reason.
