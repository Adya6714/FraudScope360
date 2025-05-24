from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class NLPModule:
    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1,2))
        self.model = LogisticRegression()

    def fit(self, texts):
        X = self.vec.fit_transform(texts)
        y = np.array([1 if "fraud" in t.lower() else 0 for t in texts])
        self.model.fit(X, y)

    def score(self, text):
        X = self.vec.transform([text])
        return self.model.predict_proba(X)[0, 1]