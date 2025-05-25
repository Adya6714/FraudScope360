
import streamlit as st
import requests

# 1. Sidebar form for input
st.sidebar.header("New Transaction")
user      = st.sidebar.text_input("User ID", "user_1")
amount    = st.sidebar.number_input("Amount", value=0.0, step=0.01)
timestamp = st.sidebar.text_input("Timestamp (ISO)", "2025-05-24T12:00:00")
device    = st.sidebar.text_input("Device ID", "device_1")
ip        = st.sidebar.text_input("IP Address", "192.168.1.1")

if st.sidebar.button("Score Transaction"):
    payload = {
        "user": user,
        "amount": amount,
        "timestamp": timestamp,
        "device": device,
        "ip": ip
    }
    try:
        # 2. Call the API
        resp = requests.post("http://127.0.0.1:8000/score", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # 3. Display results
        st.write("## Risk Score:", data["risk_score"])
        st.write("### Breakdown:")
        st.json(data["breakdown"])
    except Exception as e:
        st.error(f"Error scoring transaction: {e}")

# 4. Optionally, add historical results table, charts, etc.