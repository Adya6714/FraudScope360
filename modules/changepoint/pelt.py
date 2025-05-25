import logging, ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters

logger = logging.getLogger(__name__)

class ChangePointDetector:
    def __init__(self, model_type="rbf", pen=10):
        logger.info("Initializing ChangePointDetector: model=%s pen=%s",
                    model_type, pen)
        self.model_type = model_type
        self.pen = pen

    def score(self, series):
        try:
            logger.debug("Detecting change points in series of length %d", len(series))
            algo = rpt.Pelt(model=self.model_type)
            algo.fit(series.values)
            pts = algo.predict(pen=self.pen)
            count = max(len(pts)-1, 0)
            logger.debug("Found %d change points", count)
            return count
        except BadSegmentationParameters:
            logger.warning("Change-point detection skipped (BadSegmentationParameters)")
            return 0
        except Exception:
            logger.exception("Unexpected error in ChangePointDetector")
            return 0