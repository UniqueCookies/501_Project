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

# Use file to create csr matrices
def create_matrix(lines):
    # Extract row, col, and edge value from each line
    rows = []
    cols = []
    values = []

    for line in lines:
        row, col, value = map(float, line.strip().split())
        rows.append(int(row))
        cols.append(int(col))
        values.append(value)

    # Determine matrix dimensions
    num_rows = max(rows) + 1
    num_cols = max(cols) + 1

    # Create CSR format arrays
    data = np.array(values)
    indices = np.array(cols)
    indptr = np.zeros(num_rows + 1, dtype=int)

    # Count non-zero elements per row
    for row in rows:
        indptr[row + 1] += 1

    # Cumulative sum to obtain indptr
    indptr = np.cumsum(indptr)

    # Create CSR matrix
    csr_matrix = sp.sparse.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))

    return csr_matrix

def create_diagonal(A):
    diag = A.diagonal()
    M = sp.sparse.diags(diag, format='csr')
    return M

# Create adjncy and xadj when having edge weights
def create_adj_xadj(adjacency_list):
    adjncy = []
    xadj = [0]
    for adj in adjacency_list:

        for item in adj:
            adjncy.append(item)
        end = len(adjncy)
        xadj.append(end)

    adjncy_array = np.array(adjncy)
    xadj_array = np.array(xadj)

    return adjncy_array, xadj_array


# Create edge weights, input needs to be graph laplacian
def create_edge_list (A):
    diag = A.diagonal()
    T = np.sum(diag)

    edge_list = []
    for row, col in zip(*A.nonzero()):
        value = A[row, col]
        if value == -1:
            d_i = A[row, row]
            d_j = A[col, col]
            weight = int(T - d_i * d_j)
            edge_list.append(weight)
    edge_list = np.array(edge_list)
    return edge_list