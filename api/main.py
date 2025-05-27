# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yaml
import logging
import shap

from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

# ───────────────────────────────────────────────────────────────
# CONFIG & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FraudScope360 Scoring API")

# These will get populated on startup
history_txns_df = None
feature_cols    = None
anomaly         = None
cpd             = None
network         = None
idclust         = None
nlp             = None
explainer_anom  = None
explainer_nlp   = None
w_cfg           = None
d_cfg           = None


class Transaction(BaseModel):
    user: str
    amount: float
    timestamp: str
    device: str
    ip: str
    merchant: str
    country: str


@app.on_event("startup")
def load_everything():
    global history_txns_df, feature_cols
    global anomaly, cpd, network, idclust, nlp
    global explainer_anom, explainer_nlp
    global w_cfg, d_cfg

    # 1) Load config
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    w_cfg = cfg["fusion_weights"]
    d_cfg = cfg["decision"]

    # 2) Load your cleaned, labeled history CSV
    logger.info("Loading labeled transaction history…")
    history_txns_df = pd.read_csv(
        "data/transactions_labeled.csv",
        parse_dates=["timestamp"],
        low_memory=False
    )
    history_txns_df["user"] = history_txns_df["user"].astype(str)
    logger.info("Loaded %d historical rows", len(history_txns_df))

    # 3) Build user–user graph by country co-membership
    logger.info("Building user-graph from country co-membership…")
    uc = (
        history_txns_df[["user", "country"]]
        .dropna()
        .drop_duplicates()
    )
    left  = uc.rename(columns={"user": "u1"})
    right = uc.rename(columns={"user": "u2"})
    merged = pd.merge(left, right, on="country")
    merged = merged[merged.u1 != merged.u2]
    merged["pair"] = merged.apply(lambda r: tuple(sorted((r.u1, r.u2))), axis=1)
    edges_df = (
        merged
        .drop_duplicates("pair")
        .loc[:, ["u1", "u2"]]
        .rename(columns={"u1":"src", "u2":"dst"})
    )
    if len(edges_df) > 20_000:
        logger.info("Sampling 20k edges for Node2Vec (was %d)", len(edges_df))
        edges_df = edges_df.sample(20_000, random_state=42).reset_index(drop=True)
    logger.info("User-graph edges: %d", len(edges_df))

    # 4) Extract & store feature columns
    logger.info("Extracting historical features…")
    feats_hist = extract_features(history_txns_df)
    feature_cols = feats_hist.columns.tolist()
    logger.info("Feature dimension: %d", len(feature_cols))

    # 5a) Train Anomaly detector
    an_cfg  = cfg["anomaly"]
    anomaly = AnomalyDetector(**an_cfg)
    anomaly.fit(feats_hist)

    # 5b) Train ChangePoint detector
    cp_cfg = cfg["changepoint"]
    cpd    = ChangePointDetector(model_type=cp_cfg["model"], pen=cp_cfg["pen"])

    # 5c) Train Identity clustering
    id_cfg  = cfg["identity"]
    idclust = IdentityClustering(
        txns_df=history_txns_df,
        eps=id_cfg["eps"],
        min_samples=id_cfg["min_samples"]
    )

    # 5d) Train Network embeddings
    g_cfg   = cfg["graph"]
    network = NetworkAnalyzer(
        edges_df=edges_df,
        dimensions=g_cfg["dimensions"],
        walk_length=g_cfg["walk_length"],
        num_walks=g_cfg["num_walks"],
        window=g_cfg["window"]
    )

    # 5e) Train NLP on real merchant+country “tickets”
    logger.info("Preparing NLP training corpus from history…")
    texts = (
        history_txns_df["merchant"].fillna("UNK").astype(str)
        + " "
        + history_txns_df["country"].fillna(-1).astype(str)
    ).tolist()
    nlp_cfg = cfg["nlp"]
    nlp     = NLPModule(ngram_range=tuple(nlp_cfg["ngram_range"]), C=nlp_cfg["C"])
    nlp.fit(texts)
    logger.info("Fitted NLPModule on %d historical texts", len(texts))

    # 6) Build SHAP explainers
    logger.info("Building SHAP explainer for anomaly…")
    explainer_anom = shap.KernelExplainer(
        lambda X: anomaly.model.decision_function(X),
        shap.sample(feats_hist, 50)
    )

    logger.info("Building SHAP explainer for NLP…")
    # sample 100 texts as background
    background = nlp.vec.transform(texts[:100])
    explainer_nlp = shap.KernelExplainer(
        lambda x: nlp.model.predict_proba(x)[:, 1],
        background
    )

    logger.info("Startup complete; ready to serve requests")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/score")
def score_txn(txn: Transaction):
    try:
        # 1) Append new txn onto this user’s history
        new_df = pd.DataFrame([txn.dict()])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
        new_df["user"]      = new_df["user"].astype(str)

        user_hist = history_txns_df[history_txns_df.user == txn.user]
        full      = pd.concat([user_hist, new_df], ignore_index=True)

        # 2) Extract & align features
        feats_all = extract_features(full)
        f_new     = feats_all.iloc[[-1]].reindex(columns=feature_cols, fill_value=0)

        # 3) Build NLP context string
        context = f"{txn.merchant} {txn.country}"

        # 4a) Anomaly score (clamped ≥0)
        a_raw   = float(anomaly.model.decision_function(f_new)[0])
        a_score = max(0.0, -a_raw)

        # 4b) Change-point on last 10 amounts
        cp_score = float(cpd.score(feats_all["zscore_amount"].iloc[-10:]))

        # 4c) Network score
        net_score = float(network.score(txn.user))

        # 4d) Identity score
        id_label = idclust.score(txn.user)
        id_score = 1.0 if id_label == -1 else 0.0

        # 4e) NLP score
        nl_score = float(nlp.score(context))

        # 5) SHAP explanations
        sa = explainer_anom(f_new).values[0]
        sn = explainer_nlp(nlp.vec.transform([context])).values[0]
        def top3(vals, names):
            ix = np.argsort(np.abs(vals))[-3:][::-1]
            return [(names[i], float(vals[i])) for i in ix]

        explain = {
            "anomaly": top3(sa, feature_cols),
            "nlp":     top3(sn, nlp.vec.get_feature_names_out().tolist())
        }

        # 6) Fuse into final risk score
        arr  = np.array([a_score, cp_score, net_score, id_score, nl_score])
        w    = np.array([
            w_cfg["anomaly"],      w_cfg["change_point"],
            w_cfg["network"],      w_cfg["id_cluster"],
            w_cfg["nlp"]
        ])
        risk = float((arr * w).sum())

        return {
            "risk_score": risk,
            "breakdown": {
                "anomaly":      a_score,
                "change_point": cp_score,
                "network":      net_score,
                "id_cluster":   id_score,
                "nlp":           nl_score
            },
            "explain": explain
        }

    except Exception as e:
        logger.exception("Error in /score")
        raise HTTPException(status_code=500, detail=str(e))