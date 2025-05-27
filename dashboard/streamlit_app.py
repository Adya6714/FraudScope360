# dashboard/streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import json
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Load decision thresholds from the same config.yaml your API uses
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
AUTO_BLOCK = cfg["decision"]["auto_block_threshold"]
ALERT_T    = cfg["decision"]["alert_threshold"]
REVIEW_T   = cfg["decision"]["review_threshold"]

# ─────────────────────────────────────────────────────────────────────────────
# Cache history so we only read it once
@st.cache_data
def load_history():
    df = pd.read_csv("data/transactions_labeled.csv", parse_dates=["timestamp"])
    df["user"] = df["user"].astype(str)
    return df

history = load_history()

# Precompute unique user list for dropdown
user_list = history["user"].unique().tolist()

st.title("FraudScope360 Demo Dashboard")

# 1) Sidebar: transaction inputs
st.sidebar.header("New Transaction")
user      = st.sidebar.selectbox("User ID", user_list)     # ← pick from real history
amount    = st.sidebar.number_input("Amount", value=0.0, step=0.01)
timestamp = st.sidebar.text_input("Timestamp (ISO)", "2025-06-20T14:30:00")
device    = st.sidebar.text_input("Device ID", "device_1")
ip        = st.sidebar.text_input("IP Address", "203.0.113.55")

# merchant codes from the IEEE dataset (e.g. W, C, R, H)
merchant_options = ["W", "C", "R", "H"]
merchant = st.sidebar.selectbox("Merchant Code", merchant_options, index=0)

# country codes (addr1 from IEEE) are integers; free‐type
country = st.sidebar.text_input("Country Code", "87")

if st.sidebar.button("Score Transaction"):
    # build payload for your API
    txn = {
        "user":     user,
        "amount":   amount,
        "timestamp":timestamp,
        "device":   device,
        "ip":       ip,
        "merchant": merchant,
        "country":  country
    }

    try:
        # 2) Call the API
        resp = requests.post("http://127.0.0.1:8000/score", json=txn)
        resp.raise_for_status()
        data = resp.json()

        # 3) Show raw JSON for debugging
        st.subheader("Sent Transaction")
        st.json(txn)

        # 4) Risk Score + Decision banner
        risk = data["risk_score"]
        st.markdown(f"## 🎯 Risk Score: **{risk:.3f}**")
        if   risk >= AUTO_BLOCK: st.error("🚫 **Auto-block:** Transaction flagged as fraudulent")
        elif risk >= ALERT_T:    st.warning("⚠️ **High-risk:** Please review this transaction")
        elif risk >= REVIEW_T:   st.info("🔍 **Medium-risk:** Manual review recommended")
        else:                    st.success("✅ **Low-risk:** Transaction appears legitimate")

        # 5) Module breakdown
        st.subheader(" Breakdown by Module")
        st.json(data["breakdown"])

        # 6) SHAP feature contributions
        st.subheader(" Top-3 Feature Contributions")
        for module, contribs in data["explain"].items():
            st.markdown(f"**{module.title()}**")
            df = pd.DataFrame(contribs, columns=["feature", "mean_abs_shap"])
            st.bar_chart(df.set_index("feature"))

        # 7) User transaction history & amount-over-time plot
        st.subheader("📜 User Transaction History")
        user_hist = history[history["user"] == user]
        if not user_hist.empty:
            st.dataframe(
                user_hist
                  .sort_values("timestamp", ascending=False)
                  .head(10)
                  .reset_index(drop=True)
            )
            st.markdown("### ⏳ Amount Over Time")
            st.line_chart(user_hist.set_index("timestamp")["amount"])
        else:
            st.write("No historical transactions found for this user.")

        # 8) Analyst feedback
        st.subheader("📝 Analyst Feedback")
        col1, col2 = st.columns(2)
        if col1.button("✅ Mark True Fraud"):
            with open("feedback.csv", "a") as f:
                f.write(json.dumps({**txn, **data, "label": "fraud"}) + "\n")
            st.success("Feedback saved: fraud")
        if col2.button("❌ Mark False Alert"):
            with open("feedback.csv", "a") as f:
                f.write(json.dumps({**txn, **data, "label": "legit"}) + "\n")
            st.success("Feedback saved: legit")

    except Exception as e:
        st.error(f"Error scoring transaction: {e}")