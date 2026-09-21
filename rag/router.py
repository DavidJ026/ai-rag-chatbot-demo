"""Three-stage routing - DESIGN.md §4.

    topic gate -> RAG -> general-knowledge fallback

Out-of-scope questions never reach an answer call. RAG is always tried first.
The fallback fires on two conditions, not one: nothing cleared the similarity
floor, or the RAG answer came back with no citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .answer import answer_from_documents, stream_from_knowledge
from .clients import anthropic_client
from .config import MODEL
from .embeddings import embed_query
from .store import retrieve

KINDS = ("RECIPE", "MANUAL", "BOTH", "OUT_OF_SCOPE")

REFUSAL = (
    "This bot only answers questions about your appliances and your recipes. "
    "I can't help with that one."
)

CLASSIFIER_SYSTEM = """You route questions for a chatbot that answers only about \
the user's own appliance manuals and cooking recipes.

Reply with EXACTLY ONE word and nothing else:

RECIPE - cooking, food, ingredients, recipes, meal ideas, techniques, substitutions
MANUAL - appliance operation, settings, features, error codes, cleaning, maintenance, parts
BOTH - needs both, e.g. "what temperature should I air fry chicken thighs at"
OUT_OF_SCOPE - anything else

A short follow-up that continues the previous topic takes that topic's label."""


@dataclass
class Result:
    stage: str  # "refusal" | "documents" | "general"
    kind: str
    text: str | None = None
    stream: Any = None
    citations: list = field(default_factory=list)
    units: list = field(default_factory=list)


def classify(question, history):
    context = ""
    for message in history[-2:]:
        if message.get("text"):
            context += f"{message['role']}: {message['text']}\n"

    response = anthropic_client().messages.create(
        model=MODEL,
        max_tokens=5,
        temperature=0,
        system=CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": f"{context}question: {question}"}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    token = raw.strip().upper().strip(".,:;\"'")
    for kind in KINDS:
        if token.startswith(kind):
            return kind
    # Unparseable: search everything rather than wrongly refusing. The stage 2
    # prompt refuses off-topic questions, so scope is still enforced downstream.
    return "BOTH"


def route(question, history):
    kind = classify(question, history)
    if kind == "OUT_OF_SCOPE":
        return Result(stage="refusal", kind=kind, text=REFUSAL)

    units = retrieve(embed_query(question), kind)
    if units:
        text, citations = answer_from_documents(question, units, history)
        if citations:
            return Result(
                stage="documents",
                kind=kind,
                text=text,
                citations=citations,
                units=units,
            )

    return Result(
        stage="general",
        kind=kind,
        stream=stream_from_knowledge(question, kind, history),
    )
