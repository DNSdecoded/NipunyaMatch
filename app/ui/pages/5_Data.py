import streamlit as st

from app.ui.client import UiError, delete, get
from app.ui.Home import sidebar

sidebar()
st.header("5 · Data")
st.caption("Everything in the local SQLite DB, across all jobs. Deletes are immediate.")


def _refresh() -> None:
    st.session_state.pop("chat", None)
    st.rerun()


st.subheader("Jobs")
jobs = get("/api/jobs")
if not jobs:
    st.info("No jobs stored.")
for j in jobs:
    c1, c2 = st.columns([5, 1])
    c1.markdown(f"**#{j['id']} {j['title']}** — required: {', '.join(j['required_skills']) or '—'}")
    if c2.button("Delete", key=f"job-{j['id']}"):
        try:
            r = delete(f"/api/jobs/{j['id']}")
            if st.session_state.get("job", {}).get("id") == j["id"]:
                st.session_state.pop("job", None)
            st.toast(f"Deleted job #{j['id']} and {r['deleted_analyses']} analyses")
            _refresh()
        except UiError as e:
            st.error(str(e))

st.subheader("Candidates")
rows = get("/api/candidates")
if not rows:
    st.info("No candidates stored.")
    st.stop()
st.dataframe(
    [{"ID": r["candidate_id"], "Name": r["name"], "Email": r["email"], "Years": r["years"],
      "Parse": r["extraction_method"], "Chars": r["char_count"], "Analyses": r["analyses"]}
     for r in rows],
    use_container_width=True,
)
for r in rows:
    c1, c2 = st.columns([5, 1])
    c1.markdown(f"**{r['name'] or '(no name)'}** · {r['email'] or '—'} · id {r['candidate_id']}")
    if c2.button("Delete", key=f"cand-{r['candidate_id']}"):
        try:
            delete(f"/api/candidates/{r['candidate_id']}")
            st.toast(f"Deleted candidate {r['candidate_id']}")
            _refresh()
        except UiError as e:
            st.error(str(e))

st.divider()
st.subheader("Danger zone")
confirm = st.checkbox(f"Yes, delete all {len(rows)} candidates, their resumes and analyses")
if st.button("Clear all candidates", type="primary", disabled=not confirm):
    try:
        r = delete("/api/candidates")
        st.toast(f"Deleted {r['deleted_candidates']} candidates, {r['deleted_analyses']} analyses")
        _refresh()
    except UiError as e:
        st.error(str(e))
