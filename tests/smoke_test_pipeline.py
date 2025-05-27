# tests/smoke_test_pipeline.py

import pandas as pd
import numpy as np

from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

def test_smoke_pipeline_runs():
    # 1) Load a small sample of your real, labeled transactions
    txns = pd.read_csv("data/transactions_labeled.csv", parse_dates=["timestamp"])
    txns = txns.sample(10, random_state=42).reset_index(drop=True)

    # 2) Build NLP “tickets” from merchant + country text
    tickets = (
        txns["merchant"].fillna("UNK").astype(str)
        + " "
        + txns["country"].fillna(-1).astype(str)
    ).tolist()

    # 3) Build a minimal user–user graph by country co-membership
    uc = txns[["user", "country"]].dropna().drop_duplicates()
    left  = uc.rename(columns={"user": "u1"})
    right = uc.rename(columns={"user": "u2"})
    merged = pd.merge(left, right, on="country")
    merged = merged[merged.u1 != merged.u2]
    merged["pair"] = merged.apply(lambda r: tuple(sorted((r.u1, r.u2))), axis=1)
    edges_df = (
        merged
        .drop_duplicates("pair")
        .loc[:, ["u1", "u2"]]
        .rename(columns={"u1": "src", "u2": "dst"})
        .reset_index(drop=True)
    )

    # 4) Feature extraction on the sampled txns
    feats = extract_features(txns)

    # 5) Initialize each module with default params
    an = AnomalyDetector()
    an.fit(feats)

    cp = ChangePointDetector()

    net = NetworkAnalyzer(
        edges_df=edges_df,
        dimensions=8,
        walk_length=5,
        num_walks=10,
        window=3
    )

    idc = IdentityClustering(txns_df=txns)

    nlp = NLPModule()
    nlp.fit(tickets)

    # 6) Score a few sample transactions
    sample_feats = feats.iloc[:3]
    _ = an.predict(sample_feats)
    _ = cp.score(sample_feats["zscore_amount"])
    _ = net.score(txns["user"].iloc[0])
    _ = idc.score(txns["user"].iloc[0])
    _ = nlp.score(tickets[0])

    # If we reached here with no exceptions, everything is wired up
    assert True