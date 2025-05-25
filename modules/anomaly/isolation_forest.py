import logging
from sklearn.ensemble import IsolationForest

# Create a module‐specific logger
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, contamination=0.02, random_state=42):
        logger.info("Initializing IsolationForest with contamination=%s, random_state=%s",
                    contamination, random_state)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        try:
            logger.info("Fitting anomaly model on %d samples", len(X))
            self.model.fit(X)
            logger.info("Anomaly model training complete")
        except Exception:
            logger.exception("Failed to fit AnomalyDetector")

    def predict(self, X):
        try:
            scores = -self.model.decision_function(X)
            logger.debug("Computed anomaly scores for %d samples", len(X))
            return scores
        except Exception:
            logger.exception("Anomaly prediction failed, returning zeros")
            return [0] * len(X)