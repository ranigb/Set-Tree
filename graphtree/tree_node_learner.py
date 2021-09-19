#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List, Tuple
from tree_node import tree_node


from pandas.core.dtypes.cast import soft_convert_objects
from split_criteria import split_criteria, criteria
from graph_data import graph_data
from sklearn.tree import DecisionTreeRegressor

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

    def get_attentions_indices(self):
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
        p = self
        attentions = []
        for i in range(0, self.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            attentions = p.attentions[graph_index] + attentions
        attentions = [list(range(0,g.get_number_of_nodes()))] + attentions
        self.available_attentions[graph_index] = attentions 
        return(g.get_feature_vector(self.graph_depths, attentions, self.criteria))


    def find_best_split(self):
        labels = self.target[self.active]
        self.available_attentions =  [[] for i in range(0, len(self.data))]
        data = np.array([self.get_feature_vector_for_item(i) for i in self.active])
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

    def apply_best_split(self):
        lte_node = tree_node_learner(data = self.data, \
                    active= self.active_lte, \
                    target = self.target, \
                    parent = self, \
                    graph_depths = self.graph_depths, \
                    max_attention_depth = self.max_attention_depth, \
                    criteria = self.criteria )

        gt_node = tree_node_learner(data = self.data, \
                    active= self.active_gt, \
                    target = self.target, \
                    parent = self, \
                    graph_depths = self.graph_depths, \
                    max_attention_depth = self.max_attention_depth, \
                    criteria = self.criteria )

        self.gt = gt_node
        self.lte = lte_node
        # update split attributes
        attentions_indices = self.get_attentions_indices()
        indices = self.data[0].get_index(self.feature_index, \
            [len(self.graph_depths), len(attentions_indices), len(self.criteria), self.data[0].get_number_of_features()])
        self.depth = self.graph_depths[indices[0]]
        self.attention_depth = attentions_indices[indices[1]][0]
        self.attention_index = attentions_indices[indices[1]][1]
        self.criterion = self.criteria[indices[2]]
        self.feature = indices[3]


        ## calculate attention for current node
        self.attentions = [[] for i in range(0, len(self.data))]
        for i in self.active:
            g = self.data[i]
            _, new_attention, _ = g.get_single_feature(self.feature_index, self.graph_depths, \
                self.available_attentions[i], self.criteria, self.thresh)
            self.attentions[i] = new_attention

    def fit(self, max_number_of_leafs:int = 10, min_tree_size:int = 10, min_gain:float = 0.0):
        tiny = np.finfo(float).tiny
        if (min_gain <= tiny): # this is to prevent trying to split a node with zero gain
            min_gain = tiny
        leafs = [self]
        total_gain = 0
        potential_gains = [self.find_best_split()]
        for i in range(1,max_number_of_leafs):
            index_max = np.argmax(potential_gains)
            gain = potential_gains[index_max]
            if (gain < min_gain):
                break
            leaf_to_split = leafs.pop(index_max)
            potential_gains.pop(index_max)
            leaf_to_split.apply_best_split()
            lte = leaf_to_split.lte
            gt = leaf_to_split.gt
            potential_gains += [lte.find_best_split(), gt.find_best_split()]
            leafs += [lte, gt]
            total_gain += gain

        # compute L2 error
        L2 = 0
        for l in leafs:
            labels = l.target[l.active]
            L2 += sum((l.value - labels)**2 )
        return (L2, total_gain)


    def eval(self, g:graph_data):
        attentions_cache = [[list(range(0,g.get_number_of_nodes()))]]
        histogram=np.zeros(g.get_number_of_nodes())
        p = self
        while (p.lte != None):
            attentions = []
            for a in attentions_cache:
                attentions += a

            score, new_attentions, selected_attention = g.get_single_feature(p.feature_index, p.graph_depths, attentions, p.criteria, p.thresh)
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
            self.tree_node = tree_node(None, None, -1, 0, self.value, -1, -1, self.max_attention_depth, None)
        else:
            first_active = self.active[0]
            g = self.data[first_active]
            depth_index, attention_index, aggregator_index, col_index = \
            g.get_index(self.feature_index, \
                [len(self.graph_depths), len(self.available_attentions[first_active]),\
                     len(self.criteria), g.get_number_of_features()])

            self.tree_node = tree_node(None, None, col_index,\
                self.thresh, self.value, self.graph_depths[depth_index], \
                attention_index, self.max_attention_depth, self.criteria[aggregator_index])

            self.tree_node.lte = self.lte.get_tree_node()
            self.tree_node.gt = self.gt.get_tree_node()

        return(self.tree_node) 
    
    def print(self, indent = ""):
        if (self.gt == None):
            print(indent, "-->", self.value)
        else:
            print(indent, "f%d thresh %3f depth %2d function %5s" % (self.feature, self.thresh, self.depth, self.criterion.get_name()))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")












    
    
    
# %%
