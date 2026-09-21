# RAG Chatbot POC — Spec

What the product does. Implementation lives in `DESIGN.md`.

## Goal

A personal chatbot that answers questions about my appliance manuals and my recipes,
built as a same-day proof of concept.

## Corpus

- **Appliance manuals** — 10 PDFs, ~100 pages each, containing diagrams and photos.
- **Recipes** — 100 text documents, ~2,000 words each. Personalised; they differ from
  what a search engine would return.

All content is non-sensitive and could be public.

## Behaviour

**Topic-scoped.** The bot answers questions about appliances and cooking only. Anything
else gets a refusal: *"This bot only answers questions about your appliances and your
recipes."* No answer is generated for off-topic questions.

**Corpus first.** Every in-scope question is answered from the indexed documents when they
contain the answer.

**General-knowledge fallback.** When the documents cannot answer an in-scope question, the
bot answers from Claude's general knowledge instead of refusing.

**Every answer is badged with its source**, shown above the answer text:

| Badge | Meaning |
|---|---|
| 🟢 **From your documents** | Answered from the corpus. Shows citations and, for manuals, the source page image. |
| ⚪ **From general knowledge** | Not in your documents. Manual answers additionally warn that they may not match your specific model. |
| ⚪ **Refusal** | Off-topic. |

An answer never mixes the two sources.

## Stack

- Python backend
- Streamlit frontend (single-page chat UI)
- ChromaDB for vectors, built offline and committed to the repo
- Voyage AI `voyage-3.5` for embeddings
- `claude-haiku-4-5` for all generation

## Auth

Shared-password gate via `st.session_state`, password in Streamlit secrets. A basic
barrier, not real authentication.

## Deployment

- Streamlit Community Cloud, public HTTPS URL
- API keys and password stored as encrypted secrets, never committed

## Out of scope

- Per-user access control
- Hybrid search (BM25 + vector)
- Evaluation harness
- Chunk-level permissions
- Live web search

## Known risk

The password will be shared and forwarded. Since the corpus is non-sensitive, the exposure
is **API spend**, not data: an uncapped key behind a public URL has no ceiling. A spend limit
on the Anthropic API key is required, backed by a per-session message cap in the app.
