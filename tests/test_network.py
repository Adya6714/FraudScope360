import pandas as pd
from modules.graph.networkx_node2vec import NetworkAnalyzer

def test_network_analyzer_basic():
    # Build a trivial graph: 3 users in a chain
    edges = pd.DataFrame([("u1","u2"), ("u2","u3")], columns=["src","dst"])
    analyzer = NetworkAnalyzer(edges_df=edges, dimensions=8, walk_length=5, num_walks=10, window=3)
    # Each known user should get a numeric score
    for user in ["u1","u2","u3"]:
        score = analyzer.score(user)
        assert isinstance(score, float)
    # Unknown user should get 0
    assert analyzer.score("uX") == 0