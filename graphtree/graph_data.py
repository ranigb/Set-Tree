#%%
import numpy as np
from numpy.linalg import matrix_power
from typing import Callable, List
#%%
class graph_data:
    def __init__(self, graph: np.array, features: np.array):
        n1, n2 = np.shape(graph)
        if (n1 != n2):
            raise ValueError("graph must be a square matrix")
        t1, t2 = np.shape(features)
        if (n1 != t1):
            raise ValueError("the number of rows of features does not match the number of nodes in the graph")

        self.graph = graph
        self.attributes = features

    def propagate(self, depth : int) -> np.array:
        return (np.matmul(np.linalg.matrix_power(self.graph, depth), features))

    def get_feature_vector(self, depths: List[int], attensions: List[np.array], aggregators: List[Callable[[np.array], np.generic]]) -> np.array:
        feature_list = []
        for depth in depths:
            p = self.propagate(depth)
            for attention in attensions:
                pa = p[attention, :]
                for agg in aggregators:
                    for col in pa.T:
                        feature_list.append(agg(col))
        return(np.array(feature_list))



#%%

graph = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
features = np.array([[1, 1], [1, 2], [1, 3]])
sm = lambda x : np.sum(x)
mx = lambda x : np.max(x)
gd = graph_data(graph, features)
print(gd.get_feature_vector([0,1,2], [[0,1,2], [0]], [sm, mx] ))



# %%
