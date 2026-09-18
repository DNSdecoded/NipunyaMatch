import time

import streamlit as st

from app.ui.client import UiError, get, post
from app.ui.Home import sidebar

sidebar()
st.header("2 · Processing")

if not st.session_state.get("job"):
    st.info("No job yet. Start on the Setup page.")
    st.stop()
batch_id = st.session_state.get("batch_id")
if not batch_id:
    st.info("No resumes uploaded yet for this job. Upload them on the Setup page.")
    st.stop()

slot = st.empty()
while True:
    b = get(f"/api/batches/{batch_id}")
    with slot.container():
        st.progress(b["done"] / max(b["total"], 1), text=f"{b['done']} / {b['total']} processed")
        st.caption(f"LLM calls so far: {b['llm_calls']}")
        st.dataframe(
            [{"file": f["filename"], "status": f["status"], "method": f["extraction_method"] or "",
              "error": f["error"] or ""} for f in b["files"]],
            use_container_width=True,
        )
        if b["done"] == b["total"]:
            errors = [f for f in b["files"] if f["status"] == "error"]
            if errors and len(errors) == b["total"]:
                st.error("Every resume failed. Errors above name the cause (bad PDF, quota, key).")
            else:
                st.success("Batch complete.")
            uploads = st.session_state.get("uploads", {})
            retryable = [f["filename"] for f in errors if f["filename"] in uploads]
            b1, b2 = st.columns(2)
            if b1.button("View candidates"):
                st.switch_page("pages/3_Candidates.py")
            if retryable and b2.button(f"Retry {len(retryable)} failed"):
                try:
                    job_id = st.session_state["job"]["id"]
                    payload = [("files", (n, uploads[n], "application/pdf")) for n in retryable]
                    r = post(f"/api/jobs/{job_id}/resumes", files=payload)
                    st.session_state["batch_id"] = r["batch_id"]
                    st.rerun()
                except UiError as e:
                    st.error(str(e))
            break
    time.sleep(2)
