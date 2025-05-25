import pandas as pd
import yaml
from data_ingest.simulate import simulate_edges
from data_ingest.simulate import simulate_tickets

from data_ingest.simulate import simulate

from features.feature_engineering import extract_features

from modules.anomaly.isolation_forest import AnomalyDetector
from modules.changepoint.pelt import ChangePointDetector
from modules.graph.networkx_node2vec import NetworkAnalyzer
from modules.identity.dbscan_identity import IdentityClustering
from modules.nlp.tfidf_logistic import NLPModule


def load_config(path="configs/params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fuse(scores: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Weighted fusion of module scores:
      - scores: DataFrame with one column per module
      - weights: dict mapping module_name -> numeric weight
    """
    w = pd.Series(weights).reindex(scores.columns).fillna(1.0)
    return (scores * w).sum(axis=1) / w.sum()


def main():
    cfg = load_config()

    # ── STEP A: Simulate transactions
    batch_df = simulate(
        n_users=cfg.get("simulator", {}).get("n_users", 50),
        n_txns=cfg.get("simulator", {}).get("n_txns", 500)
    )
    if batch_df.empty:
        print("No events received.")
        return

    print(f"Processing {len(batch_df)} events")

    # ── STEP B: Feature extraction
    features_df = extract_features(batch_df)

    # ── STEP C: Initialize & fit modules
    anomaly = AnomalyDetector(
        contamination=cfg.get("anomaly", {}).get("contamination", 0.02)
    )
    anomaly.fit(features_df)

    cpd = ChangePointDetector(
        model=cfg.get("changepoint", {}).get("model", "rbf")
    )

    edges_df = simulate_edges(
        n_users=cfg.get("simulator", {}).get("n_users", 50),
        n_edges=cfg.get("simulator", {}).get("n_edges", 200)
    )
    net = NetworkAnalyzer(edges_df)

    idc = IdentityClustering(batch_df)

    tickets = simulate_tickets(
        n=cfg.get("simulator", {}).get("n_tickets", 100)
    )
    labels = [1 if "fraud" in t.lower() else 0 for t in tickets]
    nlp = NLPModule()
    nlp.fit(tickets, labels)

    # ── STEP D: Score each module
    scores = pd.DataFrame(index=features_df.index)
    scores["anomaly"]   = anomaly.predict(features_df)
    scores["change_pt"] = (
        features_df["zscore_amount"]
        .rolling(window=cfg.get("changepoint", {}).get("window", 50), min_periods=1)
        .apply(lambda s: cpd.score(s), raw=False)
    )
    scores["network"]   = batch_df["user"].map(net.score)
    scores["identity"]  = batch_df["user"].map(idc.score)
    scores["nlp"]       = pd.Series(
        [nlp.predict(txt) for txt in tickets],
        index=features_df.index
    )

    # ── STEP E: Fuse & rank
    weights = cfg.get("weights", {})
    scores["risk"] = fuse(scores, weights)

    result = pd.concat([batch_df.reset_index(drop=True), scores], axis=1)
    top10 = result.nlargest(10, "risk")[['user', 'amount', 'risk']]

    print("Top 10 High-Risk Events:")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()