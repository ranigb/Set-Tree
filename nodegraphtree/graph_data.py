#%%
from xmlrpc.client import boolean
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
        self.features = features

    def propagate(self, depth : int, attention:np.array) -> np.array:
        f = np.zeros(shape=self.features.shape)
        f[attention, :] = self.features[attention, :]
        for i in range(0, depth):
            f = self.graph @ f
        return(f)

    def get_feature_vector(self, depths: List[int], 
                            attensions: List[np.array]) -> np.array:
        feature_list = []
        for depth in depths:
            for attention in attensions:
                p = self.propagate(depth, attention)
                feature_list.append(p)
        conc = np.concatenate(feature_list, axis=1)
        return(conc)

    def get_index(self, index : int, 
                        sizes:List[int]) -> List[int]:
        indices = []
        for n in range(0, len(sizes)):
            s = sizes[len(sizes) - 1 - n]
            i = index % s
            index = int((index - i) / s)
            indices.insert(0, i)
        return(indices)

    def get_number_of_nodes(self):
        return(np.shape(self.graph)[0])

    def get_number_of_features(self):
        return(np.shape(self.features)[1])

    def get_single_feature(self, index_in_feature_vector : int, 
                                depths: List[int], 
                                attensions: List[np.array], 
                                threshold: np.generic = 0) -> np.generic:

        depth_index, attention_index, col_index = \
            self.get_index(index_in_feature_vector, [len(depths), len(attensions), self.features.shape[1]])
        depth = depths[depth_index]
        attention = attensions[attention_index]
        p = self.propagate(depth, attention)
        col = p[:, col_index]
        return(col, attention)


    def get_attentions(self, index_in_feature_vector : int,
                        threshold: np.generic, 
                        depths: List[int], 
                        attensions: List[np.array]) -> List[List[int]]:

        depth_index, attention_index, col_index = \
            self.get_index(index_in_feature_vector, [len(depths), len(attensions), self.features.shape[1]])
        depth = depths[depth_index]
        attention = attensions[attention_index]
        p = self.propagate(depth, attention)
        col = p[:, col_index]
        gt_attention = [i for i in attention if (col[i] > threshold)]
        lte_attention = [i for i in attention if (col[i] <= threshold)]
        local_attention = agg.get_attention(col, threshold)
        return([gt_attention, lte_attention])

        
