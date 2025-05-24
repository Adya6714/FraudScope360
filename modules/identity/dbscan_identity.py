from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

class IdentityClustering:
    def __init__(self, txns_df):
        users = txns_df['user'].unique()
        embeddings = []
        for u in users:
            emails = [u + "@example.com"]
            sims = [fuzz.ratio(e1, e2) for e1 in emails for e2 in emails]
            embeddings.append((u, np.mean(sims)))
        self.df = pd.DataFrame(embeddings, columns=['user', 'sim'])
        self.model = DBSCAN(eps=1, min_samples=1)
        self.df['cluster'] = self.model.fit_predict(self.df[['sim']])

    def score(self, user_id):
        row = self.df[self.df['user'] == user_id]
        return row['cluster'].iloc[0] if not row.empty else -1