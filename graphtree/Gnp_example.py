#%%
import graph_data
import numpy as np
from graphtree import graphtree

np.random.seed(seed=112433)
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
gt = graphtree(max_number_of_leafs=20)
tree = gt.fit(train, np.array(target))

print("tree node created")


# %%


tree.print()
preds = np.array([tree.predict(x)[0] for x in train ])
L2 = np.sum((preds - target)**2)
print(L2, gt.train_L2, gt.train_total_gain)

# %%

# %%
