import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from networkx import connected_watts_strogatz_graph
from scipy.io import mmread
from scipy.sparse import diags


def vertex_vertex_sparse(mat):
    mat = mat.tocsr()
    length = len(mat.data)
    mat.data = np.ones(length)
    return mat  # return SCR format


# Laplacian matrix
def laplacian_matrix_sparse(mat):
    mat = mat.tocsr()
    length = mat.shape[1]
    d = mat @ np.ones(length)
    indices = [i for i in range(length)]
    D = sp.coo_matrix((d, (indices, indices)))
    laplacian = D - mat
    return laplacian

def make_graph():
    G = nx.Graph()

    # Add nodes to the graph (20 nodes in this example)
    for i in range(20):
        G.add_node(i)

    # Add edges to the graph to specify the connectivity
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (7, 8), (8, 9), (9, 0), (9, 10), (10, 11),
             (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
             (16, 17), (17, 18), (18, 19)]

    G.add_edges_from(edges)

    # Compute the Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()
    # Set the last row and column (except the main diagonal) to 0
    n = 20
    L[n - 1, :] = 0
    L[:, n - 1] = 0
    L[n - 1, n - 1] = 2
    L = L.astype(float)
    # Find the smallest eigenvalue using eigsh
    k = 1  # Number of eigenvalues to compute (the smallest one)

    # Extract the diagonal of L and create the diagonal matrix M
    diagonal_elements = L.diagonal()
    M = diags(diagonal_elements, format='csc')
    M = M.toarray()

    return L, M


def make_graph_2(n):
    # Create a random connected graph (adjust parameters as needed)
    G = connected_watts_strogatz_graph(n, 4, 0.2, seed=42)

    # Convert the NetworkX graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2

    # Calculate the degree matrix
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))

    # Calculate the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Ensure the main diagonal is positive
    np.fill_diagonal(laplacian_matrix, np.abs(laplacian_matrix.diagonal()))
    temp = laplacian_matrix[n - 1, n - 1]
    laplacian_matrix[n - 1, :] = 0
    laplacian_matrix[:, n - 1] = 0
    laplacian_matrix[n - 1, n - 1] = temp
    laplacian_matrix = laplacian_matrix.astype(float)
    # create M matrix
    diagonal_elements = laplacian_matrix.diagonal()
    M = diags(diagonal_elements, format='csc')
    M = M.toarray()

    return laplacian_matrix, M
