import pandas as pd
from data_ingest.simulate import simulate, simulate_tickets, simulate_edges
from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule

def test_smoke_pipeline_runs():
    # 1) Simulate a tiny dataset
    txns = simulate(n_users=5, n_txns=10)
    tickets = simulate_tickets(n=10)
    edges = simulate_edges(n_users=5, n_edges=5)

    # 2) Feature extraction
    feats = extract_features(txns)

    # 3) Initialize modules with default params
    an = AnomalyDetector()
    an.fit(feats)
    cp = ChangePointDetector()
    net = NetworkAnalyzer(edges_df=edges, dimensions=8, walk_length=5, num_walks=10, window=3)
    idc = IdentityClustering(txns_df=txns)
    nlp = NLPModule()

    # 4) Score one transaction each
    sample = feats.iloc[:3]
    _ = an.predict(sample)
    _ = cp.score(sample["zscore_amount"])
    _ = net.score(txns["user"].iloc[0])
    _ = idc.score(txns["user"].iloc[0])
    _ = nlp.score(tickets[0])

    # If we reached here, everything wired up
    assert True