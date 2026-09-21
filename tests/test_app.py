import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import types
from streamlit.testing.v1 import AppTest

ok, fail = [], []
def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  <- {detail}" if not cond and detail else ""))

APP = str(ROOT / "app.py")
PW = "hunter2"

def fresh():
    at = AppTest.from_file(APP, default_timeout=60)
    at.secrets["APP_PASSWORD"] = PW
    at.secrets["ANTHROPIC_API_KEY"] = "x"
    at.secrets["VOYAGE_API_KEY"] = "x"
    return at

# --- gate blocks by default
at = fresh().run()
check("app runs without error", not at.exception, str(at.exception))
check("gate shows a password field", len(at.text_input) == 1)
check("gate hides the chat input", len(at.chat_input) == 0)
body = " ".join(m.value for m in at.markdown) + " ".join(t.value for t in at.title)
check("no corpus content rendered before auth", "From your documents" not in body)

# --- wrong password
at = fresh()
at.run()
at.text_input[0].set_value("wrong")
at.button[0].click().run()
check("wrong password is rejected", len(at.error) == 1 and "Incorrect" in at.error[0].value)
check("wrong password keeps the chat hidden", len(at.chat_input) == 0)

# --- correct password
at = fresh()
at.run()
at.text_input[0].set_value(PW)
at.button[0].click().run()
check("correct password opens the chat", len(at.chat_input) == 1)
check("session counter shown", len(at.sidebar.metric) == 1)

# --- a grounded answer renders badge + citations
from rag import router as R

class FakeUnit:
    image_path = None
class FakeCite:
    title = "oven-manual - page 12"; cited_text = "set 190C for 18 minutes"; unit = FakeUnit()

def fake_route(q, hist):
    if "moon" in q:
        return R.Result(stage="refusal", kind="OUT_OF_SCOPE", text=R.REFUSAL)
    if "generic" in q:
        return R.Result(stage="general", kind="MANUAL",
                        stream=iter(["Not from your manual", " - general info."]))
    return R.Result(stage="documents", kind="MANUAL",
                    text="Press Air Fry and set 190C.", citations=[FakeCite()])

R.route = fake_route

def ask(question):
    at = fresh()
    at.run()
    at.text_input[0].set_value(PW)
    at.button[0].click().run()
    at.chat_input[0].set_value(question).run()
    return at

at = ask("how do I air fry chicken")
md = [m.value for m in at.markdown]
check("grounded answer shows the documents badge",
      any("From your documents" in m for m in md), str(md[:6]))
check("grounded answer body rendered", any("Press Air Fry" in m for m in md))
check("citation title rendered", any("oven-manual - page 12" in m for m in md))
check("no exception on grounded path", not at.exception, str(at.exception))

at = ask("tell me about generic ovens")
md = [m.value for m in at.markdown]
check("fallback shows the general-knowledge badge",
      any("From general knowledge" in m for m in md), str(md[:6]))
check("fallback badge says not in your documents",
      any("not in your documents" in m for m in md))

at = ask("how far away is the moon")
md = [m.value for m in at.markdown]
check("off-topic shows the out-of-scope badge", any("Out of scope" in m for m in md), str(md[:6]))
check("off-topic shows the refusal text",
      any("only answers questions about your appliances" in m for m in md))

# --- session cap
at = fresh()
at.run()
at.text_input[0].set_value(PW)
at.button[0].click().run()
at.session_state.messages = [{"role": "user", "text": "q", "stage": "documents",
                              "citations": []} for _ in range(20)]
at.run()
check("session cap disables the chat input", at.chat_input[0].disabled is True)
check("session cap explains itself", len(at.info) == 1 and "limit" in at.info[0].value)

print(f"\n{len(ok)} passed, {len(fail)} failed")
sys.exit(1 if fail else 0)
