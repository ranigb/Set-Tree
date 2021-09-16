#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List, Tuple
from split_criteria import split_criteria, criteria
from graph_data import graph_data
from sklearn.tree import DecisionTreeRegressor

#%%
class tree_node:
    def __init__(self,  
                    gt : "tree_node" = None, 
                    lte : "tree_node" = None,
                    thresh: np.generic = np.nan, 
                    value: np.generic = np.nan,
                    depth: int = 0, 
                    attention_source: "tree_node" = None, 
                    attention_index: int = -1,
                    criterion: split_criteria = None, 
                    ):
        self.gt = gt
        self.lte = lte
        self.criterion = criterion
        self.value = value
        self.attention = []
        self.attention_source = attention_source
        self.thresh = thresh
        self.attention_index = attention_index
        self.depth = depth

    def print(self, indent = ""):
        if (self.gt == None):
            print(indent, "-->", self.value)
        else:
            print(indent, "thresh %3f depth %2d" % (self.thresh, self.depth))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")
    



    def eval(self, graph: graph_data, histogram:np.array = None ) -> Tuple[np.generic, np.array]:
        if (self.gt == None):
            return(tuple((self.value, histogram)))
        attention = []
        if (self.attention_source == None):
            attention = list(range(graph.get_number_of_nodes()))
        else:
            attention = self.attention_source.attention[self.attention_index]
        activations = graph.propagate(self.depth)[attention]
        self.score = self.criterion.get_score(activations)
        local_attention = self.criterion.get_attention(activations, self.thresh)
        self.attention = [[attention[i] for i in local] for local in local_attention]
        if (histogram == None):
            histogram = np.zeros([graph.get_number_of_nodes()])
        histogram[attention] += 1
        if (self.score >= self.thresh):
            return(self.gt.eval(graph, histogram))
        else:
            return(self.lte.eval(graph, histogram))


        

class tree_node_learner:
    def __init__(self, data: List[graph_data], 
                    active: List[int], 
                    target: np.array,
                    parent: "tree_node_learner" = None,
                    graph_depths: List[int] = [0, 1, 2],
                    max_attention_depth: int = 2,
                    criteria: List[split_criteria] = criteria):
        self.data = data
        self.active = active
        self.lte = None
        self.gt = None
        self.parent = parent
        self.value = np.mean(target[active])
        self.graph_depths = graph_depths
        self.max_attention_depth = max_attention_depth
        self.criteria = criteria
        self.target = target
        self.attention_depth = -1
        self.attention_index = -1
        self.thresh = np.nan
        self.depth = 0
        self.criterion = criteria[0]


    def get_tree_node(self):

        if (self.attention_depth <= 0):
            attention_source_learner = None
        else:
            attention_source_learner = self
            for i in range(0, self.attention_depth):
                attention_source_learner = attention_source_learner.parent

            attention_source_learner = attention_source_learner.tree_node    
        
        if (attention_source_learner == None):
            attention_source = None
        else:
            attention_source = attention_source_learner.tree_node

        self.tree_node = tree_node(None, None, \
            self.thresh, self.value, self.depth, \
            attention_source, self.attention_index,
            self.criterion)

        if (self.lte != None):
            self.tree_node.lte = self.lte.get_tree_node()
        if (self.gt != None):
            self.tree_node.gt = self.gt.get_tree_node()

        return(self.tree_node) 

    def get_attentions_indices(self):
        g = self.data[0]
        attention_indices = [(-1, -1)] # marking the whole set
        p = self
        for i in range(0, self.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            att = [(i, j) for j in range(0, len(p.attentions[0]))]
            attention_indices += att
        return(attention_indices)



    def get_feature_vector_for_item(self, graph_index: int):
        g = self.data[graph_index]
        attentions = [list(range(0,g.get_number_of_nodes()))]
        p = self
        for i in range(0, self.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            attentions += p.attentions[graph_index]
        return(g.get_feature_vector(self.graph_depths, attentions, self.criteria))


    def find_best_split(self):
        labels = self.target[self.active]
        data = np.array([self.get_feature_vector_for_item(i) for i in self.active])
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(data, labels)
        if (len(stump.tree_.value) < 3):
            return(0)
        self.feature_index = stump.tree_.feature[0]
        self.thresh = stump.tree_.threshold[0]
        self.lte_value = stump.tree_.value[1]
        self.gt_value = stump.tree_.value[2]
        feature_values = data[:,self.feature_index]
        active_lte_local = np.where(feature_values <= self.thresh)[0].tolist()
        self.active_lte = [self.active[i] for i in active_lte_local]
        active_gt_local = np.where(feature_values > self.thresh)[0].tolist()
        self.active_gt = [self.active[i] for i in active_gt_local]
        self.potential_gain = len(active_gt_local) * self.gt_value * self.gt_value + \
                len(active_lte_local) * self.lte_value * self.lte_value - \
                len(self.active) * stump.tree_.value[0] * stump.tree_.value[0]
        return(self.potential_gain)

    def apply_best_split(self):
        lte_node = tree_node_learner(data = self.data, \
                    active= self.active_lte, \
                    target = self.target, \
                    parent = self, 
                    graph_depths = self.graph_depths, \
                    max_attention_depth = self.max_attention_depth, \
                    criteria = self.criteria )

        gt_node = tree_node_learner(data = self.data, \
                    active= self.active_gt, \
                    target = self.target, \
                    parent = self, 
                    graph_depths = self.graph_depths, \
                    max_attention_depth = self.max_attention_depth, \
                    criteria = self.criteria )

        self.gt = gt_node
        self.lte = lte_node
        attentions_indices = self.get_attentions_indices()
        indices = self.data[0].get_index(self.feature_index, \
            [len(self.graph_depths), len(attentions_indices), len(self.criteria)])
        self.depth = self.graph_depths[indices[0]]
        self.attention_depth = attentions_indices[indices[1]][0]
        self.attention_index = attentions_indices[indices[1]][1]
        self.criterion = self.criteria[indices[2]]









    
    
    
# %%
