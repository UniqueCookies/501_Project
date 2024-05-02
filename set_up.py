import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from networkx import connected_watts_strogatz_graph
from scipy.io import mmread


# n - number of nodes
# k - Each node is joined with its k nearest neighbor
def make_graph(n, k, p, tries, seed):
    # Create a random connected graph (adjust parameters as needed)
    G = connected_watts_strogatz_graph(n, k, p, tries, seed=seed)

    # Convert the NetworkX graph to an adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_array(G)

    # Sum of each colume
    diag = adjacency_matrix.sum(axis=0)

    # Calculate the degree matrix
    M = sp.sparse.diags(diag,format='csr')

    # Calculate the Laplacian matrix
    laplacian_matrix = M - adjacency_matrix

    # Set Last Colume/Row to be 0
    n = laplacian_matrix.shape[0]
    for i in range(n - 1):
        laplacian_matrix[n - 1, i] = 0
        laplacian_matrix[i, n - 1] = 0
    # Eliminate unncessary zeros
    laplacian_matrix.eliminate_zeros()

    # Prepare adjancency to correct format for pymetis
    adjacency_matrix = adj_to_list(adjacency_matrix)

    return G, adjacency_matrix, laplacian_matrix, M


# Convert Adjacency matrix into adjancency list for pymetis
def adj_to_list(adjacency_matrix):
    lil_format = sp.sparse.lil_matrix(adjacency_matrix)
    adjacency_list = [np.array(x) for x in lil_format.rows]
    return adjacency_list


def adj_to_nx_graph(adjacency_matrix):
    # Create an empty graph
    G = nx.Graph()

    # Get the number of vertices
    num_vertices = len(adjacency_matrix)

    # Add nodes to the graph
    G.add_nodes_from(range(num_vertices))

    # Iterate through the rows of the adjacency matrix
    for i, row in enumerate(adjacency_matrix):
        # Iterate through the columns of the adjacency matrix
        for j, value in enumerate(row):
            # Add an edge if there is a connection (value is non-zero)
            if value != 0:
                G.add_edge(i, j)

    return G
