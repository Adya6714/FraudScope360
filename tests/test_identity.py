import pandas as pd
from modules.identity.dbscan_identity import IdentityClustering

def test_identity_clustering_on_real_data():
    # 1) Load your labeled transaction file
    df = pd.read_csv("data/transactions_labeled.csv", usecols=["user"])
    assert not df.empty, "No users found in transactions_labeled.csv"

    # 2) Pick two real users from the data
    users = df["user"].unique()
    assert len(users) >= 2, "Need at least two distinct users for the test"
    user_a, user_b = users[0], users[1]

    # 3) Initialize clustering on the full set
    clust = IdentityClustering(txns_df=df, eps=1.0, min_samples=1)

    # 4) Known users should yield integer cluster labels
    label_a = clust.score(user_a)
    label_b = clust.score(user_b)
    assert isinstance(label_a, int), f"Expected int for {user_a}, got {type(label_a)}"
    assert isinstance(label_b, int), f"Expected int for {user_b}, got {type(label_b)}"

    # 5) An unknown user should yield -1
    assert clust.score("this_user_does_not_exist") == -1