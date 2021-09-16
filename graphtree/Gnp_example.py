#%%
import graph_data
import numpy as np
from tree_node import tree_node_learner


def get_random_gnp(n, p, label):
    a = np.random.uniform(size = [n, n])
    a = (a + a.T)/2
    adj = (a < p).astype(int)
    features = np.ones([n,1])
    g = graph_data.graph_data(adj, features, label)
    return(g)



def get_sample(n1, p1, n0, p0, count):
    sample = []
    target = []
    for i in range(0, count):
        sample.append(get_random_gnp(n1, p1, 1))
        target.append(1)
        sample.append(get_random_gnp(n0, p0, 0))
        target.append(-1)
    return(sample,target)

#%%

train,target = get_sample(10, 0.3, 10, 0.2, 100)
tree = tree_node_learner(data = train, active = list(range(0, len(train))), 
        target=np.array(target))

print("tree node created")


# %%
potential_gain = tree.find_best_split()
print(potential_gain)

tree.apply_best_split()

# %%
prediction_tree = tree.get_tree_node()

preds = [prediction_tree.eval(x)[0] for x in train]

prediction_tree.print()