import streamlit as st

from app.ui.client import UiError, post
from app.ui.Home import sidebar

sidebar()
st.header("1 · Setup")

with st.form("jd"):
    text = st.text_area("Paste the job description", height=200)
    pdf = st.file_uploader("…or upload a JD PDF", type=["pdf"])
    if st.form_submit_button("Parse job"):
        try:
            if pdf is not None:
                upload = {"file": (pdf.name, pdf.getvalue(), "application/pdf")}
                job = post("/api/jobs", files=upload)
            elif text.strip():
                job = post("/api/jobs", data={"text": text})
            else:
                st.warning("Paste a JD or upload a PDF.")
                st.stop()
            st.session_state["job"] = job
            st.success(f"Parsed: {job['title']}")
        except UiError as e:
            st.error(str(e))

job = st.session_state.get("job")
if job:
    st.subheader("Parsed requirements")
    st.write("**Required:**", ", ".join(job["required_skills"]) or "—")
    st.write("**Preferred:**", ", ".join(job["preferred_skills"]) or "—")
    years, edu = job["min_years"] or "—", job["education_level"] or "—"
    st.write(f"**Min years:** {years} · **Education:** {edu}")
    st.caption("Mis-parsed? Edit the JD text above and parse again before uploading resumes.")

    files = st.file_uploader(
        "Upload resumes (PDF, up to 50)", type=["pdf"], accept_multiple_files=True
    )
    if files and st.button(f"Process {len(files)} resume(s)"):
        try:
            r = post(f"/api/jobs/{job['id']}/resumes",
                     files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files])
            st.session_state["batch_id"] = r["batch_id"]
            st.switch_page("pages/2_Processing.py")
        except UiError as e:
            st.error(str(e))
