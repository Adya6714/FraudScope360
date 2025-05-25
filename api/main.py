# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yaml
import logging
import shap

# Simulation & feature helpers
from data_ingest.simulate import simulate, simulate_edges, simulate_tickets
from features.feature_engineering import extract_features

# Model modules
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load configuration
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# A) Prepare historical data
history_txns_df = simulate(n_users=50, n_txns=500)
# Simulate support tickets
tickets_list    = simulate_tickets(n=100)
# Simulate user graph edges
dedges_df        = simulate_edges(n_users=50, n_edges=200)

# Extract features for anomaly module
features_hist = extract_features(history_txns_df)
feature_cols  = features_hist.columns.tolist()

# B) Initialize & train modules
an_cfg = cfg["anomaly"]
anomaly = AnomalyDetector(
    contamination=an_cfg["contamination"],
    random_state=an_cfg["random_state"]
)
anomaly.fit(features_hist)
cp_cfg = cfg["changepoint"]
cpd = ChangePointDetector(
    model_type=cp_cfg["model"],
    pen=cp_cfg["pen"]
)
id_cfg = cfg["identity"]
idclust = IdentityClustering(
    txns_df=history_txns_df,
    eps=id_cfg["eps"],
    min_samples=id_cfg["min_samples"]
)
g_cfg = cfg["graph"]
network = NetworkAnalyzer(
    edges_df=dedges_df,
    dimensions=g_cfg["dimensions"],
    walk_length=g_cfg["walk_length"],
    num_walks=g_cfg["num_walks"],
    window=g_cfg["window"]
)
nlp_cfg = cfg["nlp"]
nlp = NLPModule(
    ngram_range=tuple(nlp_cfg["ngram_range"]),
    C=nlp_cfg["C"]
)
# Fit NLP on simulated tickets
nlp.fit(tickets_list)

# C) Build SHAP explainers
logger.info("Creating SHAP KernelExplainer for IsolationForest (anomaly)")
explainer_anomaly = shap.KernelExplainer(
    lambda X: anomaly.model.decision_function(X),
    shap.sample(features_hist, 50)
)
logger.info("Creating SHAP LinearExplainer for NLPModule")
explainer_nlp = shap.LinearExplainer(
    nlp.model,
    nlp.vec.transform(tickets_list)
)

# Load fusion weights and decision thresholds
w_cfg = cfg["fusion_weights"]
d_cfg = cfg["decision"]

# FASTAPI app setup
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
        # 1) Single-row DataFrame
        df = pd.DataFrame([txn.dict()])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 2) Feature extraction & alignment
        feats = extract_features(df)
        feats = feats.reindex(columns=feature_cols, fill_value=0)
        # 3) Module scores
        a_score  = float(anomaly.predict(feats)[0])
        cp_series = feats["zscore_amount"]
        cp_score = float(cpd.score(cp_series.iloc[-10:]))
        net_score = float(network.score(txn.user))
        id_score  = int(idclust.score(txn.user))
        text      = f"User {txn.user} reported issue"
        nl_score  = float(nlp.score(text))
        # 4) SHAP explanations
        shap_an_exp = explainer_anomaly(feats)
        shap_a = shap_an_exp.values[0]
        shap_n_exp = explainer_nlp(nlp.vec.transform([text]))
        shap_n = shap_n_exp.values[0]
        def top3(vals, names):
            idx = np.argsort(np.abs(vals))[-3:][::-1]
            return [(names[i], float(vals[i])) for i in idx]
        explain = {
            "anomaly": top3(shap_a, feature_cols),
            "nlp":     top3(shap_n, nlp.vec.get_feature_names_out().tolist())
        }
        # 5) Fuse scores
        weights   = np.array([w_cfg["anomaly"], w_cfg["change_point"], w_cfg["network"], w_cfg["id_cluster"], w_cfg["nlp"]])
        scores_arr= np.array([a_score, cp_score, net_score, id_score, nl_score])
        risk      = float((scores_arr * weights).sum())
        # 6) Return JSON
        return {
            "risk_score": risk,
            "breakdown": {
                "anomaly": a_score,
                "change_point": cp_score,
                "network": net_score,
                "id_cluster": id_score,
                "nlp": nl_score
            },
            "explain": explain
        }
    except Exception as e:
        logger.exception("Error in /score endpoint")
        raise HTTPException(status_code=500, detail=str(e))
