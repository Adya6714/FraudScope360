import pandas as pd
import numpy as np
import yaml

# 1. Import your ingestion/simulation helpers
from data_ingest.simulate import simulate, simulate_tickets, simulate_edges

# 2. Import feature engineering and modules
from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# A) Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract sub-configs
an_cfg  = config["anomaly"]
cp_cfg  = config["changepoint"]
g_cfg   = config["graph"]
id_cfg  = config["identity"]
nlp_cfg = config["nlp"]
w_cfg   = config["fusion_weights"]

# ─────────────────────────────────────────────────────────────────────────────
# B) Prepare “historical” data for training the modules
#    (you can swap this out with real data later)

# 1) Transactions history for anomaly & identity modules
history_txns_df = simulate(n_users=50, n_txns=500)

# 2) Support tickets for NLP
tickets_list = simulate_tickets(n=100)
# (or: tickets_list = simulate_tickets())

# 3) User graph edges for NetworkAnalysis
edges_df     = simulate_edges(n_users=50, n_edges=200)
# (or: edges_df = simulate_edges())

# ─────────────────────────────────────────────────────────────────────────────
# C) Feature extraction on history
features_hist = extract_features(history_txns_df)

feature_cols = features_hist.columns

# ─────────────────────────────────────────────────────────────────────────────
# D) Initialize & train each module with config parameters

# 1. Anomaly
anomaly = AnomalyDetector(
    contamination=an_cfg["contamination"],
    random_state=an_cfg["random_state"]
)
anomaly.fit(features_hist)

# 2. Change-Point
cpd = ChangePointDetector(
    model_type=cp_cfg["model"],
    pen=cp_cfg["pen"]
)

# 3. Network
network = NetworkAnalyzer(
    edges_df=edges_df,
    dimensions=g_cfg["dimensions"],
    walk_length=g_cfg["walk_length"],
    num_walks=g_cfg["num_walks"],
    window=g_cfg["window"]
)

# 4. Identity Clustering
idclust = IdentityClustering(
    txns_df=history_txns_df,
    eps=id_cfg["eps"],
    min_samples=id_cfg["min_samples"]
)

# 5. NLP
nlp = NLPModule(
    ngram_range=nlp_cfg["ngram_range"],
    C=nlp_cfg["C"]
)
nlp.fit(tickets_list)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# E) Simulate new transactions and extract features
new_txns_df = simulate(n_users=50, n_txns=20)
new_txns_df["timestamp"] = pd.to_datetime(new_txns_df["timestamp"])

features_new = extract_features(new_txns_df)
features_new = features_new.reindex(columns=feature_cols, fill_value=0)

# ─────────────────────────────────────────────────────────────────────────────
# F) Score each module
# … after loading config, training modules, extracting features_new …
# … after you have `features_new` and aligned columns …

# Capture the z-score series for change-point
z = features_new["zscore_amount"]

# Build the weight array once (so w_arr is available)
import numpy as np
w_arr = np.array([
    w_cfg["anomaly"],
    w_cfg["change_point"],
    w_cfg["network"],
    w_cfg["id_cluster"],
    w_cfg["nlp"]
])

# Prepare scores_df
scores_df = pd.DataFrame(index=new_txns_df.index)

# Anomaly
try:
    scores_df["anomaly"] = anomaly.predict(features_new)
    logger.debug("Anomaly scores computed")
except Exception:
    logger.exception("Anomaly detection failed; filling zeros")
    scores_df["anomaly"] = 0

# Change-point (manual loop)
try:
    cp_scores = []
    window_size = 10
    for i in range(len(z)):
        start = max(0, i - window_size + 1)
        window = z.iloc[start : i + 1]
        cp_scores.append(cpd.score(window))
    scores_df["change_point"] = cp_scores
    logger.debug("Change-point scores computed")
except Exception:
    logger.exception("Change-point detection failed; filling zeros")
    scores_df["change_point"] = 0

# Network
try:
    scores_df["network"] = new_txns_df["user"].apply(network.score)
    logger.debug("Network scores computed")
except Exception:
    logger.exception("Network analysis failed; filling zeros")
    scores_df["network"] = 0

# Identity
try:
    scores_df["id_cluster"] = new_txns_df["user"].apply(idclust.score)
    logger.debug("Identity clustering scores computed")
except Exception:
    logger.exception("Identity clustering failed; filling -1")
    scores_df["id_cluster"] = -1

# NLP
try:
    scores_df["nlp"] = new_txns_df["user"] \
        .apply(lambda u: nlp.score(f"User {u} reported issue"))
    logger.debug("NLP scores computed")
except Exception:
    logger.exception("NLP scoring failed; filling 0.0")
    scores_df["nlp"] = 0.0

# Fuse into final risk_score
scores_df["risk_score"] = (
    scores_df[["anomaly","change_point","network","id_cluster","nlp"]]
    .values * w_arr
).sum(axis=1)

# Combine with transaction info
result = pd.concat([new_txns_df, scores_df], axis=1)

# Output
logger.info(
    "Top 5 high-risk transactions:\n%s",
    result.sort_values("risk_score", ascending=False).head(5)
)
# ─────────────────────────────────────────────────────────────────────────────
# G) Combine and display
result = pd.concat([new_txns_df, scores_df], axis=1)
logger.info("Top 5 high-risk transactions:")
logger.info("\n%s", result.sort_values("risk_score", ascending=False).head(5))

import logging

# 1. Configure the root logger
logging.basicConfig(
    level=logging.INFO,                     # capture INFO and above
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)        # module-level logger