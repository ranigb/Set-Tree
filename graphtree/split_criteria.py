#%%
from operator import attrgetter
import numpy as np
from typing import Callable, List

#%%

class split_criteria:
    def __init__(self, 
            aggregator: Callable[[np.array], np.generic], 
            attention_generator: Callable[[np.array, np.generic], List[List[int]]],
            name: str
            ):

        self.aggregator = aggregator
        self.attention_generator = attention_generator
        self.name = name

    def get_score(self, activations:np.array) -> np.generic:
        if (activations.size == 0):
            return -1000
        return(self.aggregator(activations))

    def get_attention(self, activations:np.array, threshold:np.generic):
        if (activations.size == 0):
            return [[] for g in self.attention_generator(np.zeros(1), threshold )]
        
        attentions = []
        raw_attentions = self.attention_generator(activations, threshold)
        for raw in raw_attentions:
            # the numpy.where function returns a single-item tuple which is the 
            # array of indices of relevant items. The following lines convert the
            # return value of numpy.where to a list
            if (type(raw) is tuple):
                attentions.append(raw[0].tolist())
            else:
                attentions.append(raw)
        return(attentions)

    def get_name(self):
        return(self.name)


#%%

_gen_normalized = lambda x,threshold : [np.where(x >= threshold/x.size), np.where(x < threshold/x.size)]
_gen_plain = lambda x,threshold : [np.where(x >= threshold), np.where(x < threshold)]


sum = split_criteria( \
    lambda x: np.sum(x), _gen_normalized, "sum" \
    )

avg = split_criteria( \
    lambda x: np.mean(x), _gen_plain, "avg"\
    )

max = split_criteria( \
    lambda x: np.max(x), _gen_plain, "max" \
    )

min = split_criteria( \
    lambda x: np.min(x), _gen_plain, "min" \
    )


criteria = [sum, avg, max, min]
