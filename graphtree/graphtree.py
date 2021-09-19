from tree_node_learner import tree_node_learner, tree_node_learner_parameters
from tree_node import tree_node
from split_criteria import split_criteria, criteria
from graph_data import graph_data
from typing import Callable, List, Tuple
import numpy as np

class graphtree:
    def __init__(self,
                graph_depths: List[int] = [0, 1, 2],
                max_attention_depth: int = 2,
                criteria: List[split_criteria] = criteria,
                max_number_of_leafs:int = 10, 
                min_gain:float = 0.0,
                min_leaf_size:int = 10
            ):
        self.graph_depths = graph_depths
        self.max_attention_depth = max_attention_depth
        self.criteria = criteria
        self.max_number_of_leafs = max_number_of_leafs
        self.min_gain = min_gain
        self.min_leaf_size = min_leaf_size

    def fit(self, X:List[graph_data], y:np.array):
        if (len(X) != len(y)):
            raise ValueError("Size of X and y mismatch") 
        parms = tree_node_learner_parameters(\
                graph_depths = self.graph_depths,
                max_attention_depth=self.max_attention_depth,
                criteria=self.criteria,
                max_number_of_leafs=self.max_number_of_leafs,
                min_gain=self.min_gain,
                min_leaf_size=self.min_leaf_size
            )
        self.tree_learner_ = tree_node_learner(parms, list(range(0, len(X))), None)
        self.train_L2, self.train_total_gain = self.tree_learner_.fit(X, y)
        self.tree_ = self.tree_learner_.get_tree_node()
        return(self.tree_)


    def predict(self, x:graph_data):
        return(self.tree_.predict(x))

    

    