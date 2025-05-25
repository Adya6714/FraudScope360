import pandas as pd
import numpy as np
from modules.changepoint.pelt import ChangePointDetector

def test_changepoint_detector_basic():
    # Create a simple series with a clear change in the middle
    series = pd.Series([0]*50 + [5]*50)
    detector = ChangePointDetector(model_type="l2", pen=5)
    # Score should be >= 1 (at least one change point)
    cp_count = detector.score(series)
    assert isinstance(cp_count, int)
    assert cp_count >= 1