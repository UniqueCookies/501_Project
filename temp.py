import networkx as nx
import numpy as np
import set_up
import multi
import time
import scipy as sp
import matplotlib.pyplot as plt
import pymetis
MAXINTERATION = 50




total_size = 1000
k = 100
print(f"Neigher nodes are {k},so each node is {k/total_size*100}% with other nodes")
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,30)
nc =[500,150,30]


adjncy_array, xadj_array = set_up.create_adj_xadj(adj)
edge = set_up.create_edge_list(A)
print(f"Set up is done")
try:
    _, partition_info = pymetis.part_graph(300, adjncy=adjncy_array, xadj=xadj_array, eweights=edge)
except Exception as e:
    print(f"error is {e}")





