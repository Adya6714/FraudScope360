# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yaml
import shap

# Import your pipeline pieces
from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load config and initialize everything

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Feature columns captured during training
feature_cols = cfg["feature_columns"]

# Instantiate & train or load your modules (here we assume they've been trained on historical data)
anomaly = AnomalyDetector(**cfg["anomaly"])
# anomaly.fit(...)

cpd     = ChangePointDetector(**cfg["changepoint"])
network = NetworkAnalyzer(edges_df=pd.read_csv("data/edges.csv"), **cfg["graph"])
idclust = IdentityClustering(txns_df=pd.read_csv("data/history_txns.csv"), **cfg["identity"])
nlp     = NLPModule(**cfg["nlp"])
# nlp.fit(...)

# Build SHAP explainers
explainer_anomaly = shap.TreeExplainer(anomaly.model)
explainer_nlp     = shap.LinearExplainer(nlp.model, nlp.vec.transform(
    [""]  # dummy, actual vectorizer will adapt
))

# Fusion weights
w = cfg["fusion_weights"]

# ─────────────────────────────────────────────────────────────────────────────
# 2) FastAPI setup

app = FastAPI(title="FraudScope360 Scoring API")

class Transaction(BaseModel):
    user: str
    amount: float
    timestamp: str
    device: str
    ip: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/score")
def score_txn(txn: Transaction):
    try:
        # a) Build a single-row DataFrame
        df = pd.DataFrame([txn.dict()])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # b) Feature extraction and alignment
        feats = extract_features(df)
        feats = feats.reindex(columns=feature_cols, fill_value=0)
        
        # c) Module scores
        a_score   = float(anomaly.predict(feats)[0])
        cp_score  = cpd.score(feats["zscore_amount"].iloc[-10:])  # last 10 values
        net_score = float(network.score(txn.user))
        id_score  = int(idclust.score(txn.user))
        text      = f"User {txn.user} reported issue"
        nl_score  = float(nlp.score(text))
        
        # d) SHAP explanations: top-3 for anomaly and NLP
        shap_a = explainer_anomaly.shap_values(feats)[0]
        shap_n = explainer_nlp.shap_values(nlp.vec.transform([text]))[0]
        
        def top3(shap_vals, names):
            idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
            return [(names[i], float(shap_vals[i])) for i in idx]
        
        explain = {
            "anomaly": top3(shap_a, feats.columns.tolist()),
            "nlp":     top3(shap_n, nlp.vec.get_feature_names_out().tolist())
        }
        
        # e) Fuse scores
        weights = np.array([
            w["anomaly"], w["change_point"],
            w["network"], w["id_cluster"], w["nlp"]
        ])
        scores  = np.array([a_score, cp_score, net_score, id_score, nl_score])
        risk    = float((scores * weights).sum())
        
        # —— RETURN including explain —— 
        return {
            "risk_score": risk,
            "breakdown": {
                "anomaly":      a_score,
                "change_point": cp_score,
                "network":      net_score,
                "id_cluster":   id_score,
                "nlp":          nl_score
            },
            "explain": explain   # ← make sure this line is here
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))