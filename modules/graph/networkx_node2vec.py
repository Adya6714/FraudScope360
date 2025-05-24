import networkx as nx
from node2vec import Node2Vec
import numpy as np

class NetworkAnalyzer:
    def __init__(self, edges_df):
        G = nx.from_pandas_edgelist(edges_df, 'src', 'dst')
        n2v = Node2Vec(G, dimensions=32, walk_length=10, num_walks=50)
        self.model = n2v.fit(window=5)
        self.users = list(G.nodes)

    def score(self, user_id):
        if user_id in self.users:
            vec = self.model.wv[user_id]
            return np.linalg.norm(vec)
        return 0