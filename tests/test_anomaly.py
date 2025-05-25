import pandas as pd
import numpy as np
from modules.anomaly.isolation_forest import AnomalyDetector

def test_anomaly_detector_basic():
    # Create a 2D feature array with 100 rows, 3 features
    X = pd.DataFrame(np.random.randn(100, 3), columns=['f1','f2','f3'])
    model = AnomalyDetector(contamination=0.1, random_state=0)
    # Should train without exception
    model.fit(X)
    # Predict should return an array of length 100
    scores = model.predict(X)
    assert len(scores) == 100
    # Scores should be floats
    assert all(isinstance(s, (float, np.floating)) for s in scores)