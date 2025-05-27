import pandas as pd
from modules.graph.networkx_node2vec import NetworkAnalyzer

def test_network_analyzer_on_real_data():
    # 1) Load your labeled transactions and grab three distinct users
    df = pd.read_csv("data/transactions_labeled.csv")
    users = df["user"].unique()
    assert len(users) >= 3, "Need at least 3 distinct users in data/transactions_labeled.csv"
    u1, u2, u3 = users[:3]

    # 2) Build a simple chain graph among them
    edges = pd.DataFrame(
        [(u1, u2), (u2, u3)],
        columns=["src", "dst"]
    )

    # 3) Initialize the NetworkAnalyzer on that graph
    analyzer = NetworkAnalyzer(
        edges_df=edges,
        dimensions=8,
        walk_length=5,
        num_walks=10,
        window=3
    )

    # 4) Each of our three users should return a float score
    for u in (u1, u2, u3):
        score = analyzer.score(u)
        assert isinstance(score, float), f"Expected float score for {u}, got {type(score)}"

    # 5) An unknown user should get 0
    assert analyzer.score("user_not_in_graph") == 0