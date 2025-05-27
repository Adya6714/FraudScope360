import pandas as pd
import numpy as np
import itertools

from data_ingest.simulate import simulate_tickets
from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

def test_smoke_pipeline_runs():
    # 1) Load a small sample of your real, labeled transactions
    txns = pd.read_csv("data/transactions_labeled.csv", parse_dates=["timestamp"])
    # take a reproducible slice
    txns = txns.sample(10, random_state=42).reset_index(drop=True)

    # 2) For NLP we still need some text; reuse your simulate function
    tickets = simulate_tickets(n=10)

    # 3) Build a minimal synthetic edge list from the sampled users
    users = txns["user"].unique().tolist()
    # pair consecutive users into edges
    edges = pd.DataFrame(
        [(u, v) for u, v in zip(users, users[1:] + users[:1])],
        columns=["src", "dst"]
    )

    # 4) Feature extraction
    feats = extract_features(txns)

    # 5) Initialize modules with default params (or read your cfg if you prefer)
    an = AnomalyDetector()
    an.fit(feats)

    cp = ChangePointDetector()

    net = NetworkAnalyzer(
        edges_df=edges,
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

    # If we reached here with no errors, pipeline is wired up correctly
    assert True