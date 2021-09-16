#%%
import numpy as np
from typing import Callable, List

#%%

class split_criteria:
    def __init__(self, 
            aggregator: Callable[[np.array], np.generic], 
            attention_generator: Callable[[np.array, np.generic], List[int]]
            ):

        self.aggregator = aggregator
        self.attention_generator = attention_generator

    def get_score(self, activations:np.array) -> np.generic:
        if (activations.size == 0):
            return np.NINF
        return(self.aggregator(activations))

    def get_attention(self, activations:np.array, thereshold):
        if (activations.size == 0):
            return []
        attention = self.attention_generator(activations, thereshold)
        # the numpy.where function returns a single-item tuple which is the 
        # array of indices of relevant items. The following lines convert the
        # return value of numpy.where to a list
        if (type(attention) is tuple):
            attention = attention[0]
        if (type(attention) is np.array):
            attention = attention.tolist()

        return(attention)


#%%

sum = split_criteria( \
    lambda x: np.sum(x), \
    lambda x,threshold : np.where(x >= threshold/x.size) \
    )

avg = split_criteria( \
    lambda x: np.mean(x), \
    lambda x,threshold : np.where(x >= threshold) \
    )

max = split_criteria( \
    lambda x: np.max(x), \
    lambda x,threshold : np.where(x >= threshold) \
    )

min = split_criteria( \
    lambda x: np.min(x), \
    lambda x,threshold : np.where(x < threshold) \
    )


criteria = [sum, avg, max, min]
