from tree_node_learner import tree_node_learner, tree_node_learner_parameters
from graph_data import graph_data
from typing import List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from collections import defaultdict
################################################################
###                                                          ###
### Note that line 124 of boosting.py in starboost should    ###
### be changed to:                                           ###
###   y_pred[:, i] += self.learning_rate * direction[:, i]   ###
###                                                          ###
### The same fix should be applied in line 179               ###
### /usr/local/lib/python3.8/dist-packages/starboost/        ###
###                                                          ###
################################################################

class graphtree(BaseEstimator, RegressorMixin):
    def __init__(self,
                graph:graph_data,
                graph_depths: List[int] = [0, 1, 2],
                max_attention_depth: int = 2,
                max_number_of_leafs:int = 10, 
                min_gain:float = 0.0,
                min_leaf_size:int = 10,
            ):
        self.graph_depths = graph_depths
        self.max_attention_depth = max_attention_depth
        self.max_number_of_leafs = max_number_of_leafs
        self.min_gain = min_gain
        self.min_leaf_size = min_leaf_size
        self.graph = graph

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def fit(self, X:List[int], y:np.array):#, eval_set = None):
        X = X.flatten()
        y = y.flatten()
        if (len(X) != len(y)):
            raise ValueError("Size of X and y mismatch") 
        parms = tree_node_learner_parameters(\
                graph_depths = self.graph_depths,
                max_attention_depth=self.max_attention_depth,
                max_number_of_leafs=self.max_number_of_leafs,
                min_gain=self.min_gain,
                min_leaf_size=self.min_leaf_size,
                graph=self.graph
            )
        self.tree_learner_ = tree_node_learner(parms, list(range(0, self.graph.get_number_of_nodes())), None)
        self.train_L2, self.train_total_gain = self.tree_learner_.fit(X, y)
        return(self)


    def predict(self, X:List[int]):
        all_predictions = self.tree_learner_.predict_all()
        if (isinstance(X,np.ndarray)):
            X = X[0].tolist()
        array = np.array(all_predictions[X])
        return(array.reshape(-1, 1))

    def print(self):
        self.tree_learner.print()

    

    