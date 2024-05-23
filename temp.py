import networkx as nx
import numpy as np
import set_up
import multi
import time
import scipy as sp
import matplotlib.pyplot as plt
import pymetis
from scipy.sparse import coo_matrix
MAXINTERATION = 50




total_size = 1000
k = 3
print(f"Neigher nodes are {k},so each node is {k/total_size*100}% with other nodes")
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,30)
nc =[400,40,4]

ac = A
for i in range(len(nc)):
    # Update P matrix
    coarse_size = nc[i]
    print(f"Ac shape before p is {ac.shape}")
    p = multi.coarse_matrix(adj, ac, coarse_size, True)
    p = multi.update_coarse_p(p, nc, i)
    p = sp.sparse.csr_matrix(p)  # make sure it is sparse format

    ac = p.T @ (ac @ p)

    ac = sp.sparse.csr_matrix(ac)

    adjacency, degree_matrix = multi.generate_adjacency(ac)  # create correct format of adjacency matrix for graph
    adj = set_up.adj_to_list(adjacency)
print(ac)