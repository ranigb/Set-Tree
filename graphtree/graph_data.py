#%%
from xmlrpc.client import boolean
import numpy as np
from numpy.linalg import matrix_power
from typing import Callable, List
from split_criteria import criteria, split_criteria



#%%
class graph_data:
    def __init__(self, graph: np.array, features: np.array, label: np.generic = np.NAN):
        n1, n2 = np.shape(graph)
        if (n1 != n2):
            raise ValueError("graph must be a square matrix")
        t1, t2 = np.shape(features)
        if (n1 != t1):
            raise ValueError("the number of rows of features does not match the number of nodes in the graph")

        self.graph = graph
        self.features = features
        self.label = label

    def propagate(self, depth : int) -> np.array:
        return (np.matmul(np.linalg.matrix_power(self.graph, depth), self.features))

    def get_feature_vector(self, depths: List[int], 
                            attensions: List[np.array], 
                            aggregators: List[split_criteria]) -> np.array:
        feature_list = []
        for depth in depths:
            p = self.propagate(depth)
            for attention in attensions:
                pa = p[attention, :]
                for agg in aggregators:
                    for col in pa.T:
                        feature_list.append(agg.get_score(col))
        return(np.array(feature_list))

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
                                aggregators: List[split_criteria],
                                threshold: np.generic = 0) -> np.generic:

        depth_index, attention_index, aggregator_index, col_index = \
            self.get_index(index_in_feature_vector, [len(depths), len(attensions), len(aggregators), self.features.shape[1]])
        depth = depths[depth_index]
        attention = attensions[attention_index]
        agg = aggregators[aggregator_index]
        p = self.propagate(depth)
        pa = p[attention, :]
        col = pa[:, col_index]
        return(agg.get_score(col), agg.get_attention(col, threshold), attention)


    def get_attentions(self, index_in_feature_vector : int,
                        threshold: np.generic, 
                        depths: List[int], 
                        attensions: List[np.array], 
                        aggregators: List[split_criteria]) -> List[List[int]]:

        depth_index, attention_index, aggregator_index, col_index = \
            self.get_index(index_in_feature_vector, [len(depths), len(attensions), len(aggregators), self.features.shape[1]])
        depth = depths[depth_index]
        attention = attensions[attention_index]
        agg = aggregators[aggregator_index]
        p = self.propagate(depth)
        pa = p[attention, :]
        col = pa[:, col_index]
        local_attention = agg.get_attention(col, threshold)
        return([[attention[i] for i in local] for local in local_attention])

    def get_label(self):
        return (self.label)

        
