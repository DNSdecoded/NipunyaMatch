import streamlit as st

from app.ui.client import get, health


def sidebar() -> None:
    h = health()
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
