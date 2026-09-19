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


BANDS = [("0-49", 0, 49), ("50-74", 50, 74), ("75-100", 75, 100)]
REC_ORDER = ["Shortlist", "Consider", "Reject"]


def _dashboard(job: dict[str, object]) -> None:
    rows = get(f"/api/jobs/{job['id']}/candidates")
    st.subheader(f"#{job['id']} {job['title']}")
    st.caption(
        f"Required: {', '.join(job['required_skills']) or '—'} · "
        f"Preferred: {', '.join(job['preferred_skills']) or '—'}"
    )
    if not rows:
        st.info("No resumes scored yet for this job.")
        if st.button("Upload resumes"):
            st.switch_page("pages/1_Setup.py")
        return
    recs = {r: sum(1 for x in rows if x["recommendation"] == r) for r in REC_ORDER}
    k = st.columns(4)
    k[0].metric("Candidates", len(rows))
    k[1].metric("🟢 Shortlist", recs["Shortlist"])
    k[2].metric("🟡 Consider", recs["Consider"])
    k[3].metric("🔴 Reject", recs["Reject"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Score distribution**")
        dist = {label: sum(1 for x in rows if lo <= x["final_score"] <= hi)
                for label, lo, hi in BANDS}
        st.bar_chart(dist, x_label="final score", y_label="candidates")
    with c2:
        st.markdown("**Required skills coverage**")
        cov = {x["name"] or f"#{x['candidate_id']}": x["matched"] for x in rows[:10]}
        st.bar_chart(cov, x_label="candidate", y_label="skills matched", horizontal=True)

    st.markdown("**Top 5**")
    st.dataframe(
        [{"Name": x["name"], "Score": x["final_score"], "Rec": x["recommendation"],
          "Matched": x["matched"], "Missing": x["missing"], "Years": x["years"]}
         for x in rows[:5]],
        width="stretch", hide_index=True,
    )
    b = st.columns(4)
    if b[0].button("➕ Upload more"):
        st.switch_page("pages/1_Setup.py")
    if b[1].button("📋 All candidates"):
        st.switch_page("pages/3_Candidates.py")
    if b[2].button("💬 Ask the assistant"):
        st.switch_page("pages/4_Assistant.py")
    if b[3].button("🗄️ Data"):
        st.switch_page("pages/5_Data.py")


def _empty() -> None:
    st.markdown("Screen resumes against a job description with an auditable hybrid score.")
    steps = [
        ("1 · Setup", "Paste or upload a job description; required/preferred skills are parsed."),
        ("2 · Upload", "Drop resume PDFs. Each is parsed, extracted and scored in the background."),
        ("3 · Review", "Ranked table, component breakdown, evidence-linked skills, export."),
        ("4 · Ask", "Plain-English questions answered from stored data with cited sources."),
    ]
    for col, (title, body) in zip(st.columns(4), steps, strict=True):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)
    if st.button("Start on the Setup page", type="primary"):
        st.switch_page("pages/1_Setup.py")


if __name__ == "__main__":  # pages import sidebar(); only the Home script renders here
    st.set_page_config(page_title="NipunyaMatch", layout="wide")
    sidebar()
    st.title("NipunyaMatch")
    active = st.session_state.get("job")
    if active:
        _dashboard(active)
    else:
        _empty()
