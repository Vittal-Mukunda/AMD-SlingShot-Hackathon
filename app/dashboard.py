import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="SlingShot Dashboard", layout="wide")
st.title("AMD SlingShot Hackathon - Agentic Project Manager")

st.sidebar.header("Simulation Controls")
step_hours = st.sidebar.slider("Hours to simulate", 0.5, 8.0, 1.0, 0.5)
if st.sidebar.button("Step Simulation"):
    with st.spinner("Simulating..."):
        requests.post(f"{API_URL}/simulate_step", params={"hours": step_hours})
    st.rerun()

try:
    response = requests.get(f"{API_URL}/state")
    if response.status_code == 200:
        state = response.json()
    else:
        st.error("Error fetching state")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("Failed to connect to API backend. Ensure it is running on port 8000 (uvicorn app.main:app).")
    st.stop()

st.metric("Current Simulation Time", f"{state.get('current_time', 0.0):.1f} hours")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Workers")
    workers = state.get('workers', [])
    if workers:
        df_workers = pd.DataFrame(workers)
        st.dataframe(df_workers, use_container_width=True)
    else:
        st.info("No workers initialized.")

with col2:
    st.subheader("Tasks")
    tasks = state.get('tasks', [])
    if tasks:
        df_tasks = pd.DataFrame(tasks)
        st.dataframe(df_tasks, use_container_width=True)
    else:
        st.info("No tasks initialized.")

st.divider()
st.subheader("Analytical Insights")

col_a, col_b = st.columns(2)

with col_a:
    if st.button("Predict Deadline Risks"):
        risk_res = requests.get(f"{API_URL}/predict_risk").json()
        risks = risk_res.get("risky_tasks", [])
        if risks:
            st.warning(f"Found {len(risks)} tasks at risk:")
            st.dataframe(pd.DataFrame(risks), use_container_width=True)
        else:
            st.success("No tasks currently at risk of missing deadlines.")

with col_b:
    if st.button("Optimize Resource Allocation"):
        opt_res = requests.get(f"{API_URL}/optimize").json()
        suggestions = opt_res.get("suggestions", [])
        if suggestions:
            st.info("Suggested Assignments:")
            st.dataframe(pd.DataFrame(suggestions), use_container_width=True)
        else:
            st.success("No pending tasks require assignment suggestions.")
