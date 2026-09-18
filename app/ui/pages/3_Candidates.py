import streamlit as st

from app.ui.client import API, UiError, get, post
from app.ui.Home import sidebar

sidebar()
st.header("3 · Candidates")
job = st.session_state.get("job")
if not job:
    st.info("No job yet. Start on the Setup page.")
    st.stop()

t1, t2 = st.columns([1, 4])
if not st.session_state.get("demo_mode") and t1.button(
    "Rescore all (no LLM)", help="Recompute deterministic parts with current weights"
):
    try:
        r = post(f"/api/jobs/{job['id']}/rescore")
        st.toast(f"Rescored {r['rescored']} candidates")
        st.rerun()
    except UiError as e:
        st.error(str(e))
t2.caption("Rescore = weights/aliases only. Re-analyze (per candidate) = fresh LLM judgement.")

c1, c2, c3 = st.columns(3)
min_score = c1.slider("Min score", 0, 100, 0)
skill = c2.text_input("Required skill filter")
min_years = c3.number_input("Min years", 0.0, 50.0, 0.0)
rows = get(f"/api/jobs/{job['id']}/candidates", min_score=min_score, skill=skill or None)
rows = [r for r in rows if (r["years"] or 0) >= min_years]
if not rows:
    st.info("No candidates yet. Upload resumes on the Setup page.")
    st.stop()

COLOR = {"Shortlist": "🟢", "Consider": "🟡", "Reject": "🔴"}
st.dataframe(
    [{"Name": r["name"], "Score": r["final_score"],
      "Rec": f"{COLOR[r['recommendation']]} {r['recommendation']}",
      "Matched": r["matched"], "Missing": r["missing"], "Years": r["years"],
      "Parse": r["extraction_method"]} for r in rows],
    use_container_width=True,
)
st.markdown(f"[Export CSV]({API}/api/jobs/{job['id']}/export?format=csv) · "
            f"[Export XLSX]({API}/api/jobs/{job['id']}/export?format=xlsx)")

for r in rows:
    with st.expander(f"{r['name']} — {r['final_score']} ({r['recommendation']})"):
        d = get(f"/api/candidates/{r['candidate_id']}")
        if st.button("Re-analyze (fresh LLM call)", key=f"re-{r['candidate_id']}"):
            try:
                with st.spinner("Asking the model again…"):
                    cid = r["candidate_id"]
                    post(f"/api/candidates/{cid}/reanalyze", params={"job_id": job["id"]})
                st.toast(f"Re-analyzed {r['name']}")
                st.rerun()
            except UiError as e:
                st.error(str(e))
        st.write(d["summary"])
        st.bar_chart(d["components"])
        a, b = st.columns(2)
        a.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in d["strengths"]))
        b.markdown("**Weaknesses**\n" + "\n".join(f"- {w}" for w in d["weaknesses"]))
        st.markdown("**Skills** (evidence quoted from resume text; ❌ = not found in resume)")
        for m in d["skill_matches"]:
            mark = "✅" if m["present"] else "❌"
            req = " *(required)*" if m["required"] else ""
            ev = f" — _{m['evidence']}_" if m["evidence"] else ""
            st.markdown(f"{mark} {m['skill']}{req}{ev}")
        st.markdown("**Interview questions**\n"
                    + "\n".join(f"1. {q}" for q in d["interview_questions"]))
        st.caption(f"Reasoning: {d['reasoning']}")
        st.caption(f"Parse: {d['extraction_method']}, {d['char_count']} chars"
                   + (f" · flags: {', '.join(d['flags'])}" if d["flags"] else ""))
