from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.02):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        # Higher score = more anomalous
        return -self.model.decision_function(X)
    