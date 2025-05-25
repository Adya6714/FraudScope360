# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yaml

# ❶ Create the FastAPI instance
app = FastAPI(title="FraudScope360 Scoring API")

# ❷ Define your Pydantic model
class Transaction(BaseModel):
    user: str
    amount: float
    timestamp: str
    device: str
    ip: str

# ❸ (Example) A simple health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ❹ Your /score endpoint (fill in with your real pipeline code)
@app.post("/score")
def score_txn(txn: Transaction):
    # (pseudo-code; replace with your actual logic)
    try:
        # e.g., df = pd.DataFrame([txn.dict()]); … feature extraction & scoring …
        risk = 0.5
        breakdown = {"anomaly": 0.1, "change_point": 0.0, "network": 0.2, "id_cluster": 0, "nlp": 0.2}
        return {"risk_score": risk, "breakdown": breakdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))