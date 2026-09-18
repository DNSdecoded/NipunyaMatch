import streamlit as st

from app.ui.client import get, health


def sidebar() -> None:
    h = health()
    demo = bool(h.get("demo_mode"))
    st.session_state["demo_mode"] = demo
    if demo:
        st.sidebar.warning(
            "Public demo — bring your own Gemini key. Data is shared and resets on restart."
        )
    if demo or st.session_state.get("api_key"):
        st.session_state["api_key"] = st.sidebar.text_input(
            "Your Gemini API key", type="password", key="api_key_input",
            value=st.session_state.get("api_key", ""),
            help="Free at https://aistudio.google.com/apikey — kept only in this browser session.",
        ).strip()
        if not st.session_state["api_key"]:
            st.sidebar.caption("Enter a key to parse jobs, upload resumes, or ask questions.")
    dot = "🟢" if h["gemini_configured"] else "🔴"
    extra = " + OpenRouter" if h["openrouter_configured"] else ""
    st.sidebar.markdown(f"{dot} Provider: Gemini{extra}")
    st.sidebar.caption(f"Breaker: {h['breaker_state']}")
    job = st.session_state.get("job")
    try:
        jobs = get("/api/jobs")
    except Exception:
        jobs = []
    if jobs:
        ids = [j["id"] for j in jobs]
        current = job["id"] if job and job["id"] in ids else ids[0]
        picked = st.sidebar.selectbox(
            "Active job", ids, index=ids.index(current), key="active_job",
            format_func=lambda i: f"#{i} {next(j['title'] for j in jobs if j['id'] == i)}",
        )
        if not job or job["id"] != picked:
            st.session_state["job"] = job = next(j for j in jobs if j["id"] == picked)
            st.session_state.pop("chat", None)
        count = len(get(f"/api/jobs/{job['id']}/candidates"))
        st.sidebar.markdown(f"**Candidates:** {count}")
    else:
        st.sidebar.info("No job yet. Start on the Setup page.")
    if h["breaker_state"] == "down":
        st.sidebar.error("API not reachable. Run `make dev`.")
    elif not h["gemini_configured"]:
        st.sidebar.error("GEMINI_API_KEY not configured. Set it in .env and restart the API.")


if __name__ == "__main__":  # pages import sidebar(); only the Home script renders here
    st.set_page_config(page_title="NipunyaMatch", layout="wide")
    sidebar()
    st.title("NipunyaMatch")
    st.write("1. Setup a job → 2. Upload resumes → 3. Review candidates → 4. Ask questions.")
