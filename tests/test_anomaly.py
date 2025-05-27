import pandas as pd
import numpy as np
from modules.anomaly.isolation_forest import AnomalyDetector
from features.feature_engineering import extract_features

def test_anomaly_detector_on_real_data():
    # 1) Load your labeled transactions
    df = pd.read_csv("data/transactions_labeled.csv", parse_dates=["timestamp"])
    # 2) Build the feature matrix exactly as your pipeline does
    feats = extract_features(df)
    assert not feats.empty, "Feature extraction returned no rows"

    # 3) Initialize & train the model
    model = AnomalyDetector(contamination=0.1, random_state=0)
    model.fit(feats)

    # 4) Score on the same data
    scores = model.predict(feats)

    # 5) Verify output shape and types
    assert len(scores) == len(feats), "Expected one score per row"
    assert all(isinstance(s, (float, np.floating)) for s in scores), "All scores must be floats"