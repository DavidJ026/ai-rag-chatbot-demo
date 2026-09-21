"""Claude calls for both answering stages - DESIGN.md §4.2, §4.3, §4.5.

Retrieved units are passed as document blocks with citations enabled, never
pasted into a text block. Citations do two jobs: they show the reader what is
grounded, and their absence is what routes the question to stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clients import anthropic_client
from .config import ANSWER_MAX_TOKENS, MODEL

HISTORY_TURNS = 6

RAG_SYSTEM = """You answer questions about the user's own appliance manuals and \
recipes, using only the documents provided with the question.

Rules:
- Answer only from the provided documents. If they do not contain the answer, say
  so plainly. Do not fill the gap from general knowledge.
- Reproduce quantities, temperatures, times and settings exactly as written.
  Never paraphrase a number.
- If several recipes match, present them separately, one after another, each with
  its name. Never merge ingredients or steps from different recipes into one set
  of instructions.
- Stay within appliances and cooking.
- Be concise and practical. No preamble."""

FALLBACK_SYSTEM = """You answer questions about home appliances and cooking from \
general knowledge.

The user has a personal library of appliance manuals and recipes. It does not
contain the answer to this question, so you are the fallback.

Rules:
- Make clear at the start that this is not from their documents.
- Do not invent specifics about the user's particular model - exact button names,
  menu paths, error codes or preset temperatures. Speak in general terms and say
  when something varies by model.
- If the question is not about appliances or cooking, reply only that this bot
  answers questions about the user's appliances and recipes, and answer nothing
  else.
- Be concise and practical. No preamble."""

MANUAL_DISCLAIMER = """

This question is about an appliance. Begin your answer with exactly this line,
followed by a blank line:

Not from your manual - this is general information about this type of appliance \
and may not match your model."""


@dataclass
class Citation:
    title: str
    cited_text: str
    unit: object


def _history(history):
    messages = []
    for message in history[-HISTORY_TURNS:]:
        if message.get("text"):
            messages.append({"role": message["role"], "content": message["text"]})
    return messages


def answer_from_documents(question, units, history):
    """Stage 1. Not streamed: the fallback decision depends on whether the
    finished response carries citations, so nothing can be shown until it does.

    Returns (text, [Citation]). An empty citation list means ungrounded, which
    the router treats as a fallback trigger.
    """
    content = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": unit.text},
            "title": unit.title,
            "citations": {"enabled": True},
        }
        for unit in units
    ]
    content.append({"type": "text", "text": question})

    response = anthropic_client().messages.create(
        model=MODEL,
        max_tokens=ANSWER_MAX_TOKENS,
        system=RAG_SYSTEM,
        messages=[*_history(history), {"role": "user", "content": content}],
    )

    text, citations = "", []
    for block in response.content:
        if block.type != "text":
            continue
        text += block.text
        for citation in getattr(block, "citations", None) or []:
            index = getattr(citation, "document_index", None)
            if index is None or index >= len(units):
                continue
            citations.append(
                Citation(
                    title=getattr(citation, "document_title", None) or units[index].title,
                    cited_text=getattr(citation, "cited_text", ""),
                    unit=units[index],
                )
            )
    return text, citations


def stream_from_knowledge(question, kind, history):
    """Stage 2. Streamed - there is no downstream decision to wait for."""
    system = FALLBACK_SYSTEM
    if kind in ("MANUAL", "BOTH"):
        system += MANUAL_DISCLAIMER

    with anthropic_client().messages.stream(
        model=MODEL,
        max_tokens=ANSWER_MAX_TOKENS,
        system=system,
        messages=[*_history(history), {"role": "user", "content": question}],
    ) as stream:
        yield from stream.text_stream
