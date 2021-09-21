from graph_data import graph_data
import numpy as np
from scipy.linalg import block_diag
from typing import Final

class Gnp:
    def __init__(self, n0, p0, n1, p1):
        self.n = [n0,n1]
        self.p = [p0,p1]


    name: Final = "Gnp"

    def get_gnp(self, label):
        n = self.n[label]
        p = self.p[label]
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        features = np.ones([n,1])
        g = graph_data(adj, features, label)
        return(g)



    def get_sample(self, count):
        sample = []
        y = np.random.choice([0,1], size=(count))
        X = [self.get_gnp(y_) for y_ in y]
        return(X, y)


class Gnp2:

    def __init__(self, max_size, p):
        self.max_size = max_size
        self.p = p

    name: Final = "Gnp2"


    def get_gnp(self, size):
        n = size
        p = self.p
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        return(adj)

    def get_2gnp(self, size1, size2):
        adj1 = self.get_gnp(size1)
        adj2 = self.get_gnp(size2)
        return(block_diag(adj1, adj2))

    
    def get_graph(self, size, label):
        if (size < 2):
            raise ValueError("The size of the graph cannot be smaller than 2")
        size1 = size2 = 0
        while (size1 == 0 | size2 ==0):
            size1 = np.random.binomial(size, 0.5)
            size2 = size - size1
        adj = self.get_2gnp(size1, size2)
        features = np.ones((size,2))
        features[0:size1, 1] = 0
        if (label <= 0):
            features[:, 1] = np.random.permutation(features[:, 1])
        return (graph_data(adj,features,label))
        
        

    def get_sample(self, count):
        sample = []
        y = np.random.choice([0,1], size=(count))
        sizes = np.random.binomial(size=(count), p = 0.5, n = self.max_size)
        X = [self.get_graph(sizes[i], y[i]) for i in range(0, count)]
        return(X, y)

