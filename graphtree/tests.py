#%%
from graph_data import graph_data
import numpy as np
from scipy.linalg import block_diag
from typing import Final
from networkx.generators.random_graphs import watts_strogatz_graph, barabasi_albert_graph
from networkx.linalg.graphmatrix import adjacency_matrix

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


class GnpMax:
    def __init__(self):
        pass

    name: Final = "GnpMax"

    def get_gnp(self,n, p):
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        features = np.ones([n,1])
        label = (np.max(np.sum(adj,1))>= 0.75*n).astype(int)
        g = graph_data(adj, features, label)
        return(g)

    def get_sample(self, count):
        sizes = np.random.binomial(size=(count), p = 0.5, n = 30)
        X = [self.get_gnp(sizes[i], 0.5) for i in range(0, count)]
        y = [g.label for g in X]
        return(X,y)


class BA_vs_Watts_Strogatz:
    name: Final = "BA vs Watts Strogatz"

    def get_WS(self, n, k, beta):
        g = watts_strogatz_graph(n, k, beta)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,1])
        g = graph_data(adj, features, 1)
        return(g)

    def get_BA(self, n, m):
        g = barabasi_albert_graph(n, m)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,1])
        while(np.sum(adj) < 4 * n):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if (i == j):
                continue
            if (adj[i,j]> 0):
                continue
            adj[i,j] = 1
            adj[j,i] = 1


        g = graph_data(adj, features, 0)
        return(g)

    def get_graph(self, size, label):
        if (label > 0):
            return(self.get_WS(size, 4, 0.1))
        else:
            return(self.get_BA(size, 2))


    def get_sample(self, count):
        y = np.random.choice([0,1], size=(count))
        sizes = np.random.binomial(size=(count), p = 0.5, n = 30)
        X = [self.get_graph(sizes[i], y[i]) for i in range(0, count)]
        return(X,y)


class BAmax:
    name: Final = "BA max"

    def get_BA(self, n, m, label):
        g = barabasi_albert_graph(n, m)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,2])
        if (label > 0):
            d = np.array(np.sum(adj, 1)).flatten()
            ind = np.argpartition(d, -4)[-4:]
            features[ind, 1] = 0
        else:
            ind = np.random.choice(range(0,n), size=(4), replace = False)
            features[ind, 1] = 0
        g = graph_data(adj, features, label)
        return(g)


    def get_sample(self, count):
        y = np.random.choice([0,1], size=(count))
        sizes = np.random.binomial(size=(count), p = 0.5, n = 30)
        X = [self.get_BA(sizes[i], 2, y[i]) for i in range(0, count)]
        return(X,y)


    


class BAone:
    name: Final = "BA one"

    def get_BA(self, n, m):
        g = barabasi_albert_graph(n, m)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,2])
        features[:,1] = np.random.random(size=(n))
        if (features[0,1] > 0.5):
            label = 1
        else:
            label = 0
        g = graph_data(adj, features, label)
        return(g)


    def get_sample(self, count):
        sizes = np.random.binomial(size=(count), p = 0.5, n = 30)
        X = [self.get_BA(sizes[i], 2) for i in range(0, count)]
        y = [g.label for g in X]
        return(X,y)

class GnpMaxFeature:
    def __init__(self):
        pass

    name: Final = "GnpMaxFeature"

    def get_gnp(self,n, p):
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        features = np.ones([n,2])
        features[:,1] = np.random.random(size=(n))
        d = np.array(np.sum(adj, 1)).flatten()
        k = int(n/2)
        ind = np.argpartition(d, -k)[-k:]
        m = np.mean(features[ind,1])

        label = (m>= 0.5).astype(int)
        g = graph_data(adj, features, label)
        return(g)

    def get_sample(self, count):
        sizes = np.random.binomial(size=(count), p = 0.5, n = 50)
        X = [self.get_gnp(sizes[i], 0.5) for i in range(0, count)]
        y = [g.label for g in X]
        return(X,y)


class Gnp1Q:
    def __init__(self):
        self.name = self.__class__.__name__

    
    def get_gnp(self,n, p):
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        features = np.ones([n,3])
        features[:,1] = np.random.choice([-1,1], size=(n))
        features[:,2] = np.random.choice([-1,1], size=(n))
        n = np.matmul(adj, features[:,0])
        d1 = np.matmul(adj, features[:,1])
        same_color = np.multiply(d1, features[:,2])
        label = np.any(same_color == n).astype(int)

        g = graph_data(adj, features, label)
        return(g)

    def get_sample(self, count):
        sizes = np.random.binomial(size=(count), p = 0.5, n = 100)
        X = [self.get_gnp(sizes[i], 0.3) for i in range(0, count)]
        y = [g.label for g in X]
        return(X,y)


#%%
class Gnp2Q:
    def __init__(self):
        self.name = self.__class__.__name__

    
    def get_gnp(self,n, p):
        a = np.random.uniform(size = [n, n])
        a = (a + a.T)/2
        adj = (a < p).astype(int)
        features = np.ones([n,3])
        features[:,1] = np.random.choice([-1,1], size=(n))
        features[:,2] = np.random.choice([-1,1], size=(n))
        n = np.matmul(adj, features[:,0])
        d1 = np.matmul(adj, features[:,1])
        d2 = np.matmul(adj, features[:,1])
        same_color = np.multiply(d1, d2)
        n_square = np.multiply(n, n)
        d1_n = (np.sum(d1 == n) > 3)
        d2_n = (np.sum(d2 == n) == 0)
        label = (d1_n | d2_n).astype(int)

        g = graph_data(adj, features, label)
        return(g)

    def get_sample(self, count):
        X = [self.get_gnp(50, 0.25) for _ in range(0, count)]
        y = [g.label for g in X]
        return(X,y)




#%%
#g =Gnp2Q()
#_,Y = g.get_sample(1000)
#print(np.sum(Y))


# %%
