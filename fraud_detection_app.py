# fraud_detection_app.py

import pandas as pd
import numpy as np
import yaml
import shap
import logging

from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load config
with open("configs/config.yaml", "r") as f:
    cfg      = yaml.safe_load(f)
an_cfg    = cfg["anomaly"]
cp_cfg    = cfg["changepoint"]
g_cfg     = cfg["graph"]
id_cfg    = cfg["identity"]
nlp_cfg   = cfg["nlp"]
w_cfg     = cfg["fusion_weights"]
d_cfg     = cfg["decision"]

# ─────────────────────────────────────────────────────────────────────────────
# A) REAL historical data
logger.info("Loading real history from CSV…")
history = pd.read_csv(
    "data/transactions_labeled.csv",
    parse_dates=["timestamp"],
    low_memory=False
)
history["timestamp"] = pd.to_datetime(history["timestamp"])
history["user"]      = history["user"].astype(str)
logger.info("History rows: %d", len(history))

# B) Build user-graph (country co-membership)
logger.info("Building user graph from history…")
uc    = history[["user", "country"]].dropna().drop_duplicates()
left  = uc.rename(columns={"user": "u1"})
right = uc.rename(columns={"user": "u2"})
pairs = pd.merge(left, right, on="country").query("u1 != u2")
pairs["pair"] = pairs.apply(lambda r: tuple(sorted((r.u1, r.u2))), axis=1)
edges = (
    pairs
    .drop_duplicates("pair")
    .loc[:, ["u1", "u2"]]
    .rename(columns={"u1": "src", "u2": "dst"})
)
if len(edges) > 20000:
    edges = edges.sample(20000, random_state=42).reset_index(drop=True)
logger.info("Graph edges: %d", len(edges))

# C) Extract features on real history
logger.info("Extracting features on history…")
feats_hist   = extract_features(history)
feature_cols = feats_hist.columns.tolist()
logger.info("Feature dim: %d", len(feature_cols))

# D) Initialize & train modules on real data
logger.info("Training AnomalyDetector…")
anomaly = AnomalyDetector(**an_cfg)
anomaly.fit(feats_hist)

logger.info("Initializing ChangePointDetector…")
cpd = ChangePointDetector(model_type=cp_cfg["model"], pen=cp_cfg["pen"])

logger.info("Initializing IdentityClustering…")
idclust = IdentityClustering(
    txns_df=history,
    eps=id_cfg["eps"],
    min_samples=id_cfg["min_samples"]
)

logger.info("Initializing NetworkAnalyzer…")
network = NetworkAnalyzer(
    edges_df=edges,
    dimensions=g_cfg["dimensions"],
    walk_length=g_cfg["walk_length"],
    num_walks=g_cfg["num_walks"],
    window=g_cfg["window"]
)

# E) Build NLP “tickets” from real history
logger.info("Building NLP tickets from history…")
tickets = (
    history["merchant"].fillna("UNK").astype(str)
    + " in country "
    + history["country"].fillna(-1).astype(str)
).tolist()
nlp = NLPModule(ngram_range=tuple(nlp_cfg["ngram_range"]), C=nlp_cfg["C"])
nlp.fit(tickets)

# F) SHAP explainers
logger.info("Building SHAP explainers…")
explainer_anom = shap.KernelExplainer(
    lambda X: anomaly.model.decision_function(X),
    shap.sample(feats_hist, 50)
)
explainer_nlp = shap.LinearExplainer(
    nlp.model,
    nlp.vec.transform(tickets)
)

# ─────────────────────────────────────────────────────────────────────────────
# G) “New” transactions: sample 20 from history
logger.info("Sampling new txns from real history…")
new_txns = history.sample(20, random_state=1).reset_index(drop=True)

# H) Score them
scores = []
for _, txn in new_txns.iterrows():
    user_id = str(txn.user)

    # 1) build full user series + this txn
    uhist = history[history.user == user_id]
    new_row = txn.to_frame().T.copy()
    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"])
    full   = pd.concat([uhist, new_row], ignore_index=True)

    # 2) extract & align features
    feats = extract_features(full).reindex(columns=feature_cols, fill_value=0)
    fn    = feats.iloc[[-1]]

    # 3) prepare NLP context
    ctx = (
        f"User {user_id} txn {txn.amount} at {txn.merchant} "
        f"in country {txn.country}"
    )

    # 4a) anomaly
    raw = anomaly.model.decision_function(fn)[0]
    a   = max(0.0, -float(raw))
    # 4b) change-point on last 10 z-scores
    cp = float(cpd.score(feats["zscore_amount"].iloc[-10:]))
    # 4c) network
    net = float(network.score(user_id))
    # 4d) identity
    il   = idclust.score(user_id)
    idsc = 1.0 if il == -1 else 0.0
    # 4e) nlp
    nl   = float(nlp.score(ctx))

    # fuse
    arr  = np.array([a, cp, net, idsc, nl])
    w    = np.array([
        w_cfg["anomaly"],    w_cfg["change_point"],
        w_cfg["network"],    w_cfg["id_cluster"],
        w_cfg["nlp"]
    ])
    risk = float((arr * w).sum())

    scores.append({
        **txn.to_dict(),
        "anomaly":      a,
        "change_point": cp,
        "network":      net,
        "id_cluster":   idsc,
        "nlp":           nl,
        "risk_score":   risk
    })

df_scores = pd.DataFrame(scores)
print("\nTop 5 high-risk txns:")
print(df_scores.sort_values("risk_score", ascending=False).head(5).to_string(index=False))