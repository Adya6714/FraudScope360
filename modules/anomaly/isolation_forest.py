import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, contamination=0.02, random_state=42):
        logger.info("Initializing IsolationForest(contamination=%s, random_state=%s)",
                    contamination, random_state)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        try:
            logger.info("Fitting IsolationForest on %d samples", len(X))
            self.model.fit(X)
            logger.info("IsolationForest fit complete")
        except Exception:
            logger.exception("Error fitting IsolationForest")

    def predict(self, X):
        try:
            scores = -self.model.decision_function(X)
            logger.debug("Computed anomaly scores for %d samples", len(X))
            return scores
        except Exception:
            logger.exception("Error during anomaly prediction; returning zeros")
            return [0.0] * len(X)