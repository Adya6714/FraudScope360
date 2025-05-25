import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

logger = logging.getLogger(__name__)

class NLPModule:
    def __init__(self, ngram_range=(1, 2), C=1.0):
        """
        TF-IDF + Logistic Regression for text fraud detection.
        :param ngram_range: tuple or list for TfidfVectorizer ngram_range.
        :param C: regularization strength for LogisticRegression.
        """
        logger.info("Initializing NLPModule with ngram_range=%s, C=%s", ngram_range, C)
        try:
            self.ngram_range = tuple(ngram_range)
            self.vec = TfidfVectorizer(ngram_range=self.ngram_range)
            self.model = LogisticRegression(C=C, max_iter=1000)
            logger.info("NLPModule initialization complete")
        except Exception:
            logger.exception("Failed to initialize NLPModule")

    def fit(self, texts):
        """
        Fit the TF-IDF vectorizer and logistic model.
        :param texts: list of strings.
        """
        try:
            logger.info("Fitting NLP model on %d texts", len(texts))
            X = self.vec.fit_transform(texts)
            y = np.array([1 if 'fraud' in t.lower() else 0 for t in texts])
            self.model.fit(X, y)
            logger.info("NLPModule training complete")
        except Exception:
            logger.exception("NLPModule.fit failed")

    def score(self, text):
        """
        Return probability that `text` indicates fraud.
        """
        try:
            X = self.vec.transform([text])
            prob = self.model.predict_proba(X)[0, 1]
            logger.debug("NLP score for text '%s': %.4f", text, prob)
            return prob
        except Exception:
            logger.exception("NLPModule.score failed, returning 0")
            return 0.0