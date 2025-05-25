import pandas as pd
from modules.identity.dbscan_identity import IdentityClustering

def test_identity_clustering_basic():
    # Create fake transactions with two users
    df = pd.DataFrame({"user": ["alice","bob","alice"]})
    clust = IdentityClustering(txns_df=df, eps=1.0, min_samples=1)
    # Known users get integer labels
    assert isinstance(clust.score("alice"), int)
    assert isinstance(clust.score("bob"), int)
    # Unknown user gets -1
    assert clust.score("charlie") == -1