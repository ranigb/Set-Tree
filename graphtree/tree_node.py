#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List, Tuple
from split_criteria import split_criteria
from graph_data import graph_data

#%%
class tree_node:
    def __init__(self,  
                    gte : "tree_node" = None, 
                    lt : "tree_node" = None,
                    thresh: np.generic = np.nan, 
                    value: np.generic = np.nan,
                    depth: int = 0, 
                    attention_source: "tree_node" = None, 
                    attention_index: int = -1,
                    criterion: split_criteria = None, 
                    ):
        self.gte = gte
        self.lt = lt
        self.criterion = criterion
        self.value = value
        self.attention = []
        self.attention_source = attention_source
        self.thresh = thresh
        self.attention_index = attention_index
        self.depth = depth


    def eval(self, graph: graph_data, histogram:np.array = None ) -> Tuple(np.generic, np.array):
        if (~np.isnan(self.value)):
            return((self.value, histogram))
        attention = []
        if (self.attention_source == None):
            attention = list(range(graph.get_number_of_nodes))
        else:
            attention = self.attention_source.attention[self.attention_index]
        activations = graph.propagate(self.depth)[attention]
        self.score = self.criterion.get_score(activations)
        local_attention = self.criterion.get_attention(activations, self.thresh)
        self.attention = [[attention[i] for i in local] for local in local_attention]
        if (histogram == None):
            histogram = np.array([graph.get_number_of_nodes()])
        histogram[self.attention] += 1
        if (self.score >= self.thresh):
            return(self.gte.eval(graph, histogram))
        else:
            return(self.lt.eval(graph, histogram))


        

class tree_node_learner:
    def __init__(self, data: List[graph_data], 
                    active: List[int], 
                    target: np.array,
                    parent: "tree_node_learner" = None):
        self.data = data
        self.active = active
        self.lt = None
        self.gte = None
        self.parent = parent
        self.value = np.mean(target[active])

    def get_tree_node(self):

        if (self.attention_depth <= 0):
            attention_source_learner = None
        else:
            attention_source_learner = self
            for i in range(0, self.attention_depth):
                attention_soruce = attention_source_learner.parent

            attention_soruce = attention_source_learner.tree_node    

        self.tree_node = tree_node(None, None, \
            self.thresh, self.value, self.depth, \
            attention_soruce, self.attention_index,
            self.criterion)

        if (self.lt != None):
            self.tree_node.lt = self.lt.get_tree_node()
        if (self.gte != None):
            self.tree_node.gte = self.gt.get_tree_node()

        return(self.tree_node) 

    
    
    
# %%
