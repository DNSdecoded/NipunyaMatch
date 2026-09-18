import time

import streamlit as st

from app.ui.client import get
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
            if st.button("View candidates"):
                st.switch_page("pages/3_Candidates.py")
            break
    time.sleep(2)
