import logging
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class IdentityClustering:
    def __init__(self, txns_df, eps=1.0, min_samples=1):
        """
        Clusters user identities based on string similarity.
        :param txns_df: DataFrame with a 'user' column.
        :param eps: DBSCAN eps parameter.
        :param min_samples: DBSCAN min_samples parameter.
        """
        logger.info("Initializing IdentityClustering with eps=%s, min_samples=%s", eps, min_samples)
        try:
            users = txns_df['user'].unique()
            embeddings = []
            for u in users:
                # simple self-similarity placeholder
                sims = [fuzz.ratio(u, u)]
                embeddings.append((u, np.mean(sims)))
            df = pd.DataFrame(embeddings, columns=['user', 'sim'])
            
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            df['cluster'] = self.model.fit_predict(df[['sim']])
            self.df = df
            logger.info("IdentityClustering trained on %d users, found %d clusters",
                        len(users), len(df['cluster'].unique()))
        except Exception:
            logger.exception("Failed to initialize IdentityClustering")

    def score(self, user_id):
        """
        Return the cluster label for a user, or -1 on error.
        """
        try:
            row = self.df[self.df['user'] == user_id]
            if row.empty:
                logger.warning("User %s not found in identity clusters", user_id)
                return -1
            label = int(row['cluster'].iloc[0])
            logger.debug("Identity cluster score for %s: %d", user_id, label)
            return label
        except Exception:
            logger.exception("IdentityClustering.score failed for %s", user_id)
            return -1