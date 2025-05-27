import pandas as pd
from features.feature_engineering import extract_features
from modules.changepoint.pelt import ChangePointDetector

def test_changepoint_detector_on_real_data():
    # 1) Load your labeled transactions CSV
    df = pd.read_csv("data/transactions_labeled.csv", parse_dates=["timestamp"])
    # 2) Extract exactly the same features your pipeline uses
    feats = extract_features(df)
    assert "zscore_amount" in feats.columns, "Feature extraction failed"

    # 3) Grab the first 50 z-score values as a test series
    series = feats["zscore_amount"].iloc[:50]
    assert len(series) == 50, "Unexpected series length"

    # 4) Initialize & run the detector
    detector = ChangePointDetector(model_type="rbf", pen=10)
    cp_count = detector.score(series)

    # 5) Confirm we get an integer ≥ 0
    assert isinstance(cp_count, int), "Expected integer count"
    assert cp_count >= 0, "Change-point count should be non-negative"