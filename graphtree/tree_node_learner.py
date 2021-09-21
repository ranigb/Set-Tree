#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List, Tuple, NamedTuple
from tree_node import tree_node


from pandas.core.dtypes.cast import soft_convert_objects
from split_criteria import split_criteria, criteria
from graph_data import graph_data
from sklearn.tree import DecisionTreeRegressor

class tree_node_learner_parameters(NamedTuple):
    graph_depths: List[int] = [0, 1, 2],
    max_attention_depth: int = 2,
    criteria: List[split_criteria] = criteria,
    max_number_of_leafs:int = 10, 
    min_leaf_size:int = 10, 
    min_gain:float = 0.0,

class tree_node_learner:
    def __init__(self,  
                    parms: tree_node_learner_parameters,
                    active: List[int], 
                    parent: "tree_node_learner" = None):
        self.active = active
        self.lte = None
        self.gt = None
        self.parent = parent
        self.parms = parms

    def get_attentions_indices(self):
        attention_indices = [(-1, -1)] # marking the whole set
        p = self
        for i in range(0, self.parms.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            att = [(i, j) for j in range(0, len(p.attentions[0]))]
            attention_indices += att
        return(attention_indices)



    def get_feature_vector_for_item(self, X, graph_index: int):
        g = X[graph_index]
        p = self
        attentions = []
        for _ in range(0, self.parms.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            attentions = p.attentions[graph_index] + attentions
        attentions = [list(range(0,g.get_number_of_nodes()))] + attentions
        self.available_attentions[graph_index] = attentions 
        return(g.get_feature_vector(self.parms.graph_depths, attentions, self.parms.criteria))


    def find_best_split(self, X:List[graph_data], y:np.array):
        labels = y[self.active]
        self.value = np.mean(labels)
        self.feature_dimension = X[0].get_number_of_features()
        if (len(self.active) < self.parms.min_leaf_size):
            self.potential_gain = 0.0
            return(0.0)
        self.available_attentions =  [[] for _ in range(0, len(X))]
        data = np.array([self.get_feature_vector_for_item(X, i) for i in self.active])
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(data, labels)
        if (len(stump.tree_.value) < 3):
            return(0)
        self.feature_index = stump.tree_.feature[0]
        self.thresh = stump.tree_.threshold[0]
        self.lte_value = stump.tree_.value[1][0][0]
        self.gt_value = stump.tree_.value[2][0][0]
        feature_values = data[:,self.feature_index]
        active_lte_local = np.where(feature_values <= self.thresh)[0].tolist()
        self.active_lte = [self.active[i] for i in active_lte_local]
        active_gt_local = np.where(feature_values > self.thresh)[0].tolist()
        self.active_gt = [self.active[i] for i in active_gt_local]
        self.potential_gain = len(active_gt_local) * self.gt_value * self.gt_value + \
                len(active_lte_local) * self.lte_value * self.lte_value - \
                len(self.active) * stump.tree_.value[0][0][0] * stump.tree_.value[0][0][0]
        return(self.potential_gain)

    def apply_best_split(self,X:List[graph_data] , y:np.array):
        lte_node = tree_node_learner( parms=self.parms,\
                    active= self.active_lte, \
                    parent = self)
        lte_node.value = np.mean(y[self.active_lte])

        gt_node = tree_node_learner( parms=self.parms,\
                    active= self.active_gt, \
                    parent = self)
        gt_node.value = np.mean(y[self.active_gt])

        self.gt = gt_node
        self.lte = lte_node
        # update split attributes
        attentions_indices = self.get_attentions_indices()
        indices = X[0].get_index(self.feature_index, \
            [len(self.parms.graph_depths), len(attentions_indices), len(self.parms.criteria), X[0].get_number_of_features()])
        self.depth = self.parms.graph_depths[indices[0]]
        self.attention_depth = attentions_indices[indices[1]][0]
        self.attention_index = attentions_indices[indices[1]][1]
        self.criterion = self.parms.criteria[indices[2]]
        self.feature = indices[3]


        ## calculate attention for current node
        self.attentions = [[] for _ in range(0, len(X))]
        for i in self.active:
            g = X[i]
            _, new_attention, _ = g.get_single_feature(self.feature_index, self.parms.graph_depths, \
                self.available_attentions[i], self.parms.criteria, self.thresh)
            self.attentions[i] = new_attention

    def fit(self, X: List[graph_data], y:np.array):
        tiny = np.finfo(float).tiny
        min_gain = self.parms.min_gain
        if (min_gain <= tiny): # this is to prevent trying to split a node with zero gain
            min_gain = tiny
        leafs = [self]
        total_gain = 0
        potential_gains = [self.find_best_split(X, y)]
        for _ in range(1,self.parms.max_number_of_leafs):
            index_max = np.argmax(potential_gains)
            gain = potential_gains[index_max]
            if (gain < min_gain):
                break
            leaf_to_split = leafs.pop(index_max)
            potential_gains.pop(index_max)
            leaf_to_split.apply_best_split(X, y)
            lte = leaf_to_split.lte
            gt = leaf_to_split.gt
            potential_gains += [lte.find_best_split(X,y), gt.find_best_split(X,y)]
            leafs += [lte, gt]
            total_gain += gain

        # compute L2 error
        L2 = 0
        for l in leafs:
            labels = y[l.active]
            L2 += sum((l.value - labels)**2 )
        return (L2, total_gain)


    def predict(self, g:graph_data):
        attentions_cache = [[list(range(0,g.get_number_of_nodes()))]]
        histogram=np.zeros(g.get_number_of_nodes())
        p = self
        while (p.lte != None):
            attentions = []
            for a in attentions_cache:
                attentions += a

            score, new_attentions, selected_attention = g.get_single_feature(p.feature_index, p.parms.graph_depths, attentions, p.parms.criteria, p.thresh)
            histogram[selected_attention] += 1
            if (len(attentions_cache) > p.max_attention_depth):
                attentions_cache.pop(1)
            attentions_cache.append(new_attentions)
            if (score <= p.thresh):
                p = p.lte
            else:
                p = p.gt
        return(p.value, histogram)

    def get_tree_node(self):
        if (self.gt == None): # this is a leaf node
            self.tree_node = tree_node(None, None, -1, 0, self.value, -1, -1, self.parms.max_attention_depth, None)
        else:
            first_active = self.active[0]
            g = graph_data(graph=np.zeros((1,1)), features=np.zeros((1,self.feature_dimension)))
            depth_index, attention_index, aggregator_index, col_index = \
            g.get_index(self.feature_index, \
                [len(self.parms.graph_depths), len(self.available_attentions[first_active]),\
                     len(self.parms.criteria), self.feature_dimension])

            self.tree_node = tree_node(None, None, col_index,\
                self.thresh, self.value, self.parms.graph_depths[depth_index], \
                attention_index, self.parms.max_attention_depth, self.parms.criteria[aggregator_index])

            self.tree_node.lte = self.lte.get_tree_node()
            self.tree_node.gt = self.gt.get_tree_node()

        return(self.tree_node) 
    
    def print(self, indent = ""):
        if (self.gt == None):
            print(indent, "-->", self.value)
        else:
            print(indent, "f%d thresh %3f depth %2d function %5s" % (self.feature, self.thresh, self.depth, self.parms.criterion.get_name()))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")












    
    
    
# %%
