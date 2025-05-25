import streamlit as st
import requests
import pandas as pd            
import json                    

# 1. Sidebar form for input
st.sidebar.header("New Transaction")
user      = st.sidebar.text_input("User ID", "user_1")
amount    = st.sidebar.number_input("Amount", value=0.0, step=0.01)
timestamp = st.sidebar.text_input("Timestamp (ISO)", "2025-05-24T12:00:00")
device    = st.sidebar.text_input("Device ID", "device_1")
ip        = st.sidebar.text_input("IP Address", "192.168.1.1")

if st.sidebar.button("Score Transaction"):
    # build the payload
    txn = {
        "user": user,
        "amount": amount,
        "timestamp": timestamp,
        "device": device,
        "ip": ip
    }

    try:
        # 2. Call the API
        resp = requests.post("http://127.0.0.1:8000/score", json=txn)
        resp.raise_for_status()
        data = resp.json()

        # 3. Display results
        st.write("## Risk Score:", data["risk_score"])

        # Breakdown from each module
        st.write("### Breakdown")
        st.json(data["breakdown"])

        #  Show SHAP explainability charts
        st.write("### Top-3 Feature Contributions")
        for module, contribs in data["explain"].items():
            st.write(f"**{module}**")
            # construct a small DataFrame
            df = pd.DataFrame(contribs, columns=["feature", "mean_abs_shap"])
            # use bar_chart for visualization
            st.bar_chart(df.set_index("feature"))

        # ⬇️ NEW: Analyst Feedback buttons
        st.write("### Analyst Feedback")
        col1, col2 = st.columns(2)
        if col1.button("✅ Mark True Fraud"):
            # append a JSON line with the txn, scores, explain, and label
            with open("feedback.csv", "a") as f:
                f.write(json.dumps({**txn, **data, "label": "fraud"}) + "\n")
            st.success("Feedback saved: fraud")
        if col2.button("❌ Mark False Alert"):
            with open("feedback.csv", "a") as f:
                f.write(json.dumps({**txn, **data, "label": "legit"}) + "\n")
            st.success("Feedback saved: legit")

    except Exception as e:
        st.error(f"Error scoring transaction: {e}")