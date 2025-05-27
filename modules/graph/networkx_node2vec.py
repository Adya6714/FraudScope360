# modules/graph/networkx_node2vec.py

import logging
import networkx as nx
from node2vec import Node2Vec
import numpy as np

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    def __init__(self, edges_df, dimensions=32, walk_length=10, num_walks=50, window=5):
        """
        :param edges_df: DataFrame with ['src', 'dst'] columns representing the user graph
        :param dimensions: embedding vector size
        :param walk_length: length of random walks
        :param num_walks: number of walks per node
        :param window: context size for Word2Vec
        """
        logger.info("Building graph with %d edges", len(edges_df))
        G = nx.from_pandas_edgelist(edges_df, 'src', 'dst')
        logger.info(
            "Initializing Node2Vec embedding (dimensions=%s, walk_length=%s, num_walks=%s)",
            dimensions, walk_length, num_walks
        )
        n2v = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks
        )
        self.model = n2v.fit(window=window)
        self.users = set(G.nodes)

    def score(self, user_id):
        """
        Returns the L2 norm of the node embedding for the given user_id.
        If the user is not in the graph, returns 0.0 as a Python float.
        """
        try:
            vec = self.model.wv[user_id]
            # Cast to native Python float so isinstance(..., float) is True
            score = float(np.linalg.norm(vec))
            logger.debug("Network score for %s: %f", user_id, score)
            return score
        except KeyError:
            logger.warning("User %s not found in graph; network score=0.0", user_id)
            return 0.0
        except Exception:
            logger.exception("NetworkAnalyzer.score failed for %s", user_id)
            return 0.0