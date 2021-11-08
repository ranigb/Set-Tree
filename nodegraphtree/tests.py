from graph_data import graph_data
import numpy as np
from scipy.linalg import block_diag
from typing import Final
from networkx.generators.random_graphs import watts_strogatz_graph, barabasi_albert_graph, fast_gnp_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix

class Gnp_overfit:
    def __init__(self, p):
        self.p = p

    name: Final = "Gnp_overfit"

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        y = np.random.choice([0,1], size=(n))
        features = np.ones([n,2])
        features[:,1] = y
        g = graph_data(adj, features)
        all = [x for x in range(0,n)]
        y = np.asmatrix(y)
        y = y.reshape((n,-1))
        return(g, y, all, all)




class Gnp_sign_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,2])
        features[:,1] = np.random.normal(size=(n))
        y = (0.5 * np.sign(adj @ features[:,1]) + 0.5).astype(int)
        y = y.reshape((n,-1))

        g = graph_data(adj, features)
        all = [x for x in range(0,n)]
        return(g, y, all, all)

class Gnp_sign_red_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,3])
        features[:,2] = np.random.choice([0,1], size=(n))
        features[:,1] = np.random.normal(size=(n))
        t = np.multiply(features[:,1], features[:,2])
        y = (0.5 * np.sign(adj @ t) + 0.5).astype(int)
        y = y.reshape((n,-1))

        g = graph_data(adj, features)
        all = [x for x in range(0,n)]
        return(g, y, all, all)


class Gnp_sign_red_blue_neighbor:
    def __init__(self, p):
        self.p = p
        self.name = self.__class__.__name__

    def get_graph(self, n):
        g = fast_gnp_random_graph(n, self.p)
        adj = adjacency_matrix(g).todense()
        features = np.ones([n,3])
        features[:,2] = np.random.choice([-1,1], size=(n))
        features[:,1] = np.random.normal(size=(n))
        t = np.multiply(features[:,1], features[:,2])
        y = (0.5 * np.sign(adj @ t) + 0.5).astype(int)
        y = y.reshape((n,-1))

        g = graph_data(adj, features)
        all = [x for x in range(0,n)]
        return(g, y, all, all)
