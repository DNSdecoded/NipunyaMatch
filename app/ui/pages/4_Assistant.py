import streamlit as st

from app.ui.client import UiError, post
from app.ui.Home import sidebar

sidebar()
st.header("4 · Assistant")
job = st.session_state.get("job")
if not job:
    st.info("No job yet. Start on the Setup page.")
    st.stop()

STARTERS = ["Show me the top 5 candidates", "Which candidates know Python?",
            "Which candidates are missing Docker?", "Recommend the best candidate for interview"]
chat = st.session_state.setdefault("chat", [])


def ask(q: str) -> None:
    try:
        r = post(f"/api/jobs/{job['id']}/query", json={"question": q})
        chat.append(
            {"q": q, "a": r["answer"], "sources": r["sources"], "provider": r["provider_used"]}
        )
    except UiError as e:
        chat.append({"q": q, "a": f"⚠️ {e}", "sources": [], "provider": "—"})


for col, s in zip(st.columns(len(STARTERS)), STARTERS, strict=True):
    if col.button(s):
        ask(s)

for turn in chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["sources"]:
            st.markdown(" ".join(f"`{s['name']} ({s['score']})`" for s in turn["sources"]))
        st.caption(f"provider: {turn['provider']}")

if q := st.chat_input("Ask about the candidates"):
    ask(q)
    st.rerun()
