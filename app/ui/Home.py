import streamlit as st

from app.ui.client import get, health


def sidebar() -> None:
    h = health()
    dot = "🟢" if h["gemini_configured"] else "🔴"
    extra = " + OpenRouter" if h["openrouter_configured"] else ""
    st.sidebar.markdown(f"{dot} Provider: Gemini{extra}")
    st.sidebar.caption(f"Breaker: {h['breaker_state']}")
    job = st.session_state.get("job")
    if job:
        st.sidebar.markdown(f"**Job:** {job['title']}")
        count = len(get(f"/api/jobs/{job['id']}/candidates"))
        st.sidebar.markdown(f"**Candidates:** {count}")
    else:
        st.sidebar.info("No job yet. Start on the Setup page.")
    if h["breaker_state"] == "down":
        st.sidebar.error("API not reachable. Run `make dev`.")
    elif not h["gemini_configured"]:
        st.sidebar.error("GEMINI_API_KEY not configured. Set it in .env and restart the API.")


if __name__ == "__main__" or st.runtime.exists():
    st.set_page_config(page_title="NipunyaMatch", layout="wide")
    sidebar()
    st.title("NipunyaMatch")
    st.write("1. Setup a job → 2. Upload resumes → 3. Review candidates → 4. Ask questions.")
