#%%
from operator import attrgetter
import numpy as np
from typing import List, NamedTuple
from pandas.core.dtypes.cast import soft_convert_objects
from graph_data import graph_data
from sklearn.tree import DecisionTreeRegressor


def intersect(lst1, lst2):
    if (isinstance(lst1, np.ndarray)):
        lst1 = lst1.tolist()
    if (isinstance(lst2, np.ndarray)):
        lst2 = lst2.tolist()
    return(list(set(lst1) & set(lst2)))

class tree_node_learner_parameters(NamedTuple):
    graph: graph_data = 0,
    graph_depths: List[int] = [0, 1, 2],
    max_attention_depth: int = 2,
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



    def get_feature_matrix(self):
        p = self
        attentions = []
        for _ in range(0, self.parms.max_attention_depth):
            if (p.parent == None):
                break
            p = p.parent
            attentions = p.attentions + attentions
        attentions = [list(range(0,self.parms.graph.get_number_of_nodes()))] + attentions
        return(self.parms.graph.get_feature_vector(self.parms.graph_depths, attentions), attentions)


    def find_best_split(self, X:List[int], y:np.array):
        active_train = intersect(self.active, X)
        labels = y[active_train]
        self.value = np.mean(labels)
        self.feature_dimension = self.parms.graph.get_number_of_features()
        if (len(labels) < self.parms.min_leaf_size):
            self.potential_gain = 0.0
            return(0.0)
        full_data, self.available_attentions = self.get_feature_matrix()
        train_data = full_data[active_train,:]
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(train_data, labels)
        if (len(stump.tree_.value) < 3):
            return(0)
        self.feature_index = stump.tree_.feature[0]
        self.thresh = stump.tree_.threshold[0]
        self.lte_value = stump.tree_.value[1][0][0]
        self.gt_value = stump.tree_.value[2][0][0]
        feature_values = full_data[:,self.feature_index]
        active_lte_local = np.where(feature_values <= self.thresh)[0].tolist()
        self.active_lte = intersect(self.active, active_lte_local)
        active_gt_local = np.where(feature_values > self.thresh)[0].tolist()
        self.active_gt = intersect(self.active, active_gt_local)
        self.potential_gain = len(self.active_gt) * self.gt_value * self.gt_value + \
                len(self.active_lte) * self.lte_value * self.lte_value - \
                len(self.active) * stump.tree_.value[0][0][0] * stump.tree_.value[0][0][0]


        attentions_indices = self.get_attentions_indices()
        indices = self.parms.graph.get_index(self.feature_index, \
            [len(self.parms.graph_depths), len(self.available_attentions),  self.feature_dimension])
        self.to_be_depth = self.parms.graph_depths[indices[0]]
        self.to_be_selected_attention = self.available_attentions[indices[1]] 
        self.to_be_feature = indices[2]
        self.to_be_attentions = \
            [intersect(self.to_be_selected_attention, active_gt_local), 
             intersect(self.to_be_selected_attention, active_lte_local)]

        return(self.potential_gain)

    def apply_best_split(self,X:List[int] , y:np.array):
        lte_node = tree_node_learner( parms=self.parms,\
                    active= self.active_lte, \
                    parent = self)
        lte_node.value = self.lte_value

        gt_node = tree_node_learner( parms=self.parms,\
                    active= self.active_gt, \
                    parent = self)
        gt_node.value = self.gt_value

        self.gt = gt_node
        self.lte = lte_node
        # update split attributes
        self.depth = self.to_be_depth
        self.feature = self.to_be_feature
        self.selected_attention = self.to_be_selected_attention
        self.attentions = self.to_be_attentions


    def fit(self, X: List[int], y:np.array):
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
            for i in range(0, len(X)):
                if (X[i] in l.active):
                    L2 += (l.value - y[i])**2
        return (L2, total_gain)

    def predict_all(self, predictions:np.array = None) -> np.array:
        if (predictions is None):
            predictions = np.zeros(shape=(self.parms.graph.get_number_of_nodes()))
        if (self.lte is None):
            for a in self.active:
                predictions[a] = self.value
        else:
            predictions = self.lte.predict_all(predictions)
            predictions = self.gt.predict_all(predictions)
        self.all_predictions = predictions
        return(predictions)



    def predict(self, x:int):
        if (not(hasattr(self, 'all_predictions'))):
            self.predict_all()
        return(self.all_predictions[x])

    
    def print(self, indent = ""):
        if (self.gt == None):
            print(indent, "-->", self.value)
        else:
            print(indent, "f%d thresh %3f depth %2d" % (self.feature, self.thresh, self.depth))
            self.lte.print(indent + "  ")
            self.gt.print(indent + "  ")












    
    
    
# %%
