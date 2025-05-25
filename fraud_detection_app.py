import pandas as pd
import numpy as np
import yaml
import shap
import logging

# 1. Simulation helpers
from data_ingest.simulate import simulate, simulate_tickets, simulate_edges

# 2. Feature engineering and modules
from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Sub-configs
an_cfg  = config["anomaly"]
cp_cfg  = config["changepoint"]
g_cfg   = config["graph"]
id_cfg  = config["identity"]
nlp_cfg = config["nlp"]
w_cfg   = config["fusion_weights"]
d_cfg   = config["decision"]

# ─────────────────────────────────────────────────────────────────────────────
# A) Prepare historical data
history_txns_df = simulate(n_users=50, n_txns=500)
tickets_list    = simulate_tickets(n=100)
edges_df        = simulate_edges(n_users=50, n_edges=200)

# B) Feature extraction on history
features_hist = extract_features(history_txns_df)
feature_cols  = features_hist.columns.tolist()

# C) Initialize & train modules
anomaly = AnomalyDetector(
    contamination=an_cfg["contamination"],
    random_state=an_cfg["random_state"]
)
anomaly.fit(features_hist)
logger.info("Creating SHAP explainer for AnomalyDetector")
explainer_anomaly = shap.TreeExplainer(anomaly.model)

cpd = ChangePointDetector(
    model_type=cp_cfg["model"],
    pen=cp_cfg["pen"]
)

network = NetworkAnalyzer(
    edges_df=edges_df,
    dimensions=g_cfg["dimensions"],
    walk_length=g_cfg["walk_length"],
    num_walks=g_cfg["num_walks"],
    window=g_cfg["window"]
)

idclust = IdentityClustering(
    txns_df=history_txns_df,
    eps=id_cfg["eps"],
    min_samples=id_cfg["min_samples"]
)

nlp = NLPModule(
    ngram_range=tuple(nlp_cfg["ngram_range"]),
    C=nlp_cfg["C"]
)
nlp.fit(tickets_list)
logger.info("Creating SHAP explainer for NLPModule")
explainer_nlp = shap.LinearExplainer(nlp.model, nlp.vec.transform(tickets_list))

# ─────────────────────────────────────────────────────────────────────────────
# D) Simulate new transactions and extract features
new_txns_df   = simulate(n_users=50, n_txns=20)
new_txns_df["timestamp"] = pd.to_datetime(new_txns_df["timestamp"])
features_new = extract_features(new_txns_df)
# Align features: add missing, drop extra based on history
features_new = features_new.reindex(columns=feature_cols, fill_value=0)

# ─────────────────────────────────────────────────────────────────────────────
# E) Score modules
scores_df = pd.DataFrame(index=new_txns_df.index)

# 1) Anomaly
try:
    scores_df["anomaly"] = anomaly.predict(features_new)
    logger.debug("Anomaly scores computed")
except Exception:
    logger.exception("Anomaly detection failed; filling zeros")
    scores_df["anomaly"] = 0

# 2) Change-point
z = features_new["zscore_amount"]
try:
    cp_scores = []
    window_size = 10
    for i in range(len(z)):
        start = max(0, i - window_size + 1)
        window = z.iloc[start:i+1]
        cp_scores.append(cpd.score(window))
    scores_df["change_point"] = cp_scores
    logger.debug("Change-point scores computed")
except Exception:
    logger.exception("Change-point detection failed; filling zeros")
    scores_df["change_point"] = 0

# 3) Network
try:
    scores_df["network"] = new_txns_df["user"].apply(network.score)
    logger.debug("Network scores computed")
except Exception:
    logger.exception("Network analysis failed; filling zeros")
    scores_df["network"] = 0

# 4) Identity
try:
    scores_df["id_cluster"] = new_txns_df["user"].apply(idclust.score)
    logger.debug("Identity clustering scores computed")
except Exception:
    logger.exception("Identity clustering failed; filling -1")
    scores_df["id_cluster"] = -1

# 5) NLP
try:
    scores_df["nlp"] = new_txns_df["user"].apply(
        lambda u: nlp.score(f"User {u} reported issue")
    )
    logger.debug("NLP scores computed")
except Exception:
    logger.exception("NLP scoring failed; filling 0.0")
    scores_df["nlp"] = 0.0

# 6) Fuse into final risk_score
w_arr = np.array([
    w_cfg["anomaly"], w_cfg["change_point"],
    w_cfg["network"], w_cfg["id_cluster"], w_cfg["nlp"]
])
scores_df["risk_score"] = (
    scores_df[["anomaly","change_point","network","id_cluster","nlp"]]
    .values * w_arr
).sum(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# F) Aggregate results
result = pd.concat([new_txns_df, scores_df], axis=1)
logger.info("Top 5 high-risk transactions:\n%s", result.sort_values("risk_score", ascending=False).head(5))

# ─────────────────────────────────────────────────────────────────────────────
# G) SHAP explainability
shap_anom = explainer_anomaly.shap_values(features_new)
texts_new = [f"User {u} reported issue" for u in new_txns_df["user"]]
X_nlp_new = nlp.vec.transform(texts_new)
shap_nlp   = explainer_nlp.shap_values(X_nlp_new)

def top_k(shap_vals, names, k=3):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[-k:][::-1]
    return [(names[i], float(mean_abs[i])) for i in idx]

feature_names      = feature_cols
text_feature_names = nlp.vec.get_feature_names_out().tolist()
explanations       = []
for i in range(len(new_txns_df)):
    explanations.append({
        "anomaly": top_k(shap_anom[i:i+1], feature_names),
        "nlp":     top_k(shap_nlp[i:i+1], text_feature_names)
    })
result["explain"] = explanations

# ─────────────────────────────────────────────────────────────────────────────
# H) Rule‐based Decision Layer (no Kafka, just simulate)
blocked, alerts, for_review = [], [], []
auto_block = d_cfg["auto_block_threshold"]
alert_t    = d_cfg["alert_threshold"]
review_t   = d_cfg["review_threshold"]
for _, row in result.iterrows():
    score = row["risk_score"]
    txn   = row.to_dict()
    if score >= auto_block:
        blocked.append(txn)
    elif score >= alert_t:
        alerts.append(txn)
    elif score >= review_t:
        for_review.append(txn)

print("\n=== DECISION BREAKDOWN ===")
print(f"Blocked   ({len(blocked)} txns):")
for b in blocked:
    print("  ", b)
print(f"\nAlerts    ({len(alerts)} txns):")
for a in alerts:
    print("  ", a)
print(f"\nReview    ({len(for_review)} txns):")
for r in for_review:
    print("  ", r)
