"""Shared stubs: fake Anthropic + Voyage so tests never hit the network."""
import hashlib, types

class Block:
    def __init__(self, text, citations=None):
        self.type = "text"; self.text = text; self.citations = citations

class Cite:
    def __init__(self, idx, title, quoted):
        self.type = "char_location"; self.document_index = idx
        self.document_title = title; self.cited_text = quoted

class Resp:
    def __init__(self, blocks): self.content = blocks

class Stream:
    def __init__(self, chunks): self._c = chunks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self): return iter(self._c)

class Messages:
    def __init__(self, owner): self.o = owner
    def count_tokens(self, **kw):
        text = kw["messages"][0]["content"]
        return types.SimpleNamespace(input_tokens=max(1, len(text) // 4))
    def create(self, **kw):
        self.o.calls.append(kw)
        return self.o.responder(kw)
    def stream(self, **kw):
        self.o.calls.append(kw)
        return Stream(self.o.streamer(kw))

class FakeAnthropic:
    def __init__(self, responder=None, streamer=None):
        self.calls = []
        self.responder = responder or (lambda kw: Resp([Block("ok")]))
        self.streamer = streamer or (lambda kw: ["general ", "answer"])
        self.messages = Messages(self)

class FakeVoyage:
    """Deterministic hash embeddings; identical text -> identical vector."""
    DIM = 32
    def embed(self, texts, model=None, input_type=None):
        assert input_type in ("document", "query"), f"bad input_type {input_type!r}"
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = [(h[i % len(h)] / 255.0) - 0.5 for i in range(self.DIM)]
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return types.SimpleNamespace(embeddings=out)

def make_pdf(pages):
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=612, height=792)
    b = io.BytesIO(); w.write(b); return b.getvalue()
