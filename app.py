#!/usr/bin/env python3
"""Streamlit chat UI. Read-only against the committed chroma_db/."""

import hmac

import streamlit as st

from rag.config import MAX_SESSION_MESSAGES, secret
from rag.router import route

st.set_page_config(page_title="Kitchen Assistant", page_icon="🍳")


# --- Auth gate -------------------------------------------------------------
# Must stay the first executable block (CLAUDE.md rule 7). Nothing that can
# render corpus content may run above it.
def authenticated():
    if st.session_state.get("authed"):
        return True

    st.title("🍳 Kitchen Assistant")
    st.caption("Questions about your appliances and your recipes.")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(password, str(secret("APP_PASSWORD"))):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    return False


if not authenticated():
    st.stop()


# --- Provenance badges -----------------------------------------------------
# Every answer carries one, above the text (SPEC.md, CLAUDE.md rule 15).
BADGES = {
    "documents": "🟢 **From your documents**",
    "general": "⚪ **From general knowledge** — not in your documents",
    "refusal": "⚪ **Out of scope**",
}


def render_citations(citations):
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        images = []
        for citation in citations:
            st.markdown(f"**{citation['title']}**")
            if citation.get("cited_text"):
                st.caption(f"> {citation['cited_text'].strip()}")
            path = citation.get("image_path")
            if path and path not in images:
                images.append(path)
        for path in images:
            st.image(path, caption="manual page")


def render_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(BADGES[message["stage"]])
        st.markdown(message["text"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))


# --- Chat ------------------------------------------------------------------
st.title("🍳 Kitchen Assistant")
st.caption(
    "Answers from your appliance manuals and recipes. "
    "Every answer says where it came from."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    render_message(message)

asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
remaining = MAX_SESSION_MESSAGES - asked
st.sidebar.metric("Questions left this session", max(remaining, 0))
if remaining <= 0:
    st.info(
        f"Session limit of {MAX_SESSION_MESSAGES} questions reached. "
        "Reload the page to start over."
    )

question = st.chat_input(
    "Ask about an appliance or a recipe", disabled=remaining <= 0
)

if question:
    history = [
        {"role": m["role"], "text": m["text"]} for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # The spinner covers the topic gate, retrieval and the stage 1 call.
            # Stage 1 cannot be streamed: whether it is shown at all depends on
            # whether the finished response carries citations.
            with st.spinner("Searching your documents…"):
                result = route(question, history)

            st.markdown(BADGES[result.stage])
            if result.stage == "general":
                text = st.write_stream(result.stream)
            else:
                text = result.text
                st.markdown(text)

            citations = [
                {
                    "title": c.title,
                    "cited_text": c.cited_text,
                    "image_path": c.unit.image_path,
                }
                for c in result.citations
            ]
            render_citations(citations)
        except Exception as error:  # noqa: BLE001 - surface failures in the UI
            st.error(f"Something went wrong: {error}")
            st.session_state.messages.pop()
            st.stop()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": text,
            "stage": result.stage,
            "citations": citations,
        }
    )
