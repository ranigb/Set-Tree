#%%
import graph_data
import numpy as np
from tree_node_learner import tree_node_learner

np.random.seed(seed=11233)
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

train_L2, train_gain = tree.fit(max_number_of_leafs = 200)
prediction_tree = tree.get_tree_node()
prediction_tree.print()

print ("-------------")
tree.print()
print("------------")
preds = np.array([tree.eval(x)[0] for x in train ])
L2 = np.sum((preds - target)**2)
repreds = np.array([prediction_tree.eval(x)[0] for x in train ])
reL2 = np.sum((repreds - target)**2)
print(reL2, L2, train_L2, train_gain)

# %%

# %%
