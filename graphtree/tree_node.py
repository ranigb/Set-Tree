#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List, Tuple

from pandas.core.dtypes.cast import soft_convert_objects
from split_criteria import split_criteria, criteria
from graph_data import graph_data
from sklearn.tree import DecisionTreeRegressor

#%%
class tree_node:
    def __init__(self,  
                    gt : "tree_node" = None, 
                    lte : "tree_node" = None,
                    feature: int = -1,
                    thresh: np.generic = np.nan, 
                    value: np.generic = np.nan,
                    depth: int = 0, 
                    attention_index: int = -1,
                    max_attention_depth: int = -1,
                    criterion: split_criteria = None, 
                    ):
        self.gt = gt
        self.lte = lte
        self.criterion = criterion
        self.value = value
        self.thresh = thresh
        self.attention_index = attention_index
        self.depth = depth
        self.feature = feature
        self.max_attention_depth = max_attention_depth

    def print(self, indent = ""):
        if (self.gt == None):
            print(indent, "-->", self.value)
        else:
            print(indent, "f%d thresh %3f depth %2d function %5s" % (self.feature, self.thresh, self.depth, self.criterion.get_name()))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")
    


    def eval(self, g:graph_data):
        attentions_cache = [[list(range(0,g.get_number_of_nodes()))]]
        histogram=np.zeros(g.get_number_of_nodes())
        pnt = self
        while (pnt.lte != None):
            attentions = []
            for a in attentions_cache:
                attentions += a
            p = g.propagate(pnt.depth)
            attention = attentions[pnt.attention_index]
            pa = p[attention, :]
            col = pa[:, pnt.feature]
            score = pnt.criterion.get_score(col)
            new_attentions = pnt.criterion.get_attention(col, pnt.thresh)
            selected_attention = attention
            histogram[selected_attention] += 1
            if (len(attentions_cache) > pnt.max_attention_depth):
                attentions_cache.pop(1)
            attentions_cache.append(new_attentions)

            if (score <= pnt.thresh):
                pnt = pnt.lte
            else:
                pnt = pnt.gt
        return(pnt.value, histogram)


 
