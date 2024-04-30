import math
import networkx as nx
import numpy as np
import scipy as sp
import pymetis


# Set up for the coordinate descent method
# create 2x2 matrix for each column of A and M
def a_test(v, i, A):
    A = sp.sparse.csr_matrix(A)
    vAv = v.T @ A @ v
    col = A.getcol(i)
    va1 = v.T @ col

    test = np.zeros((2, 2))
    test[0, 0] = vAv
    test[0, 1] = va1.item()
    test[1, 0] = va1
    test[1, 1] = A[i, i]
    return test


# Coordinate descent method
def update_v(v, A, M):
    for i in range(v.size):
        test_a = a_test(v, i, A)
        test_m = a_test(v, i, M)
        _, min_eigenvector = sp.linalg.eigh(test_a, test_m, subset_by_index=[0, 0])
        t = min_eigenvector[1] / min_eigenvector[0]
        v[i] += t
    norm = v.T @ M @ v
    norm = math.sqrt(norm)
    v = v / norm
    return v


###########################################################################
# find eigenvalues,vectors for the coarsest level
def find_eigen(A, M):
    eigenvalues, eigenvectors = sp.linalg.eigh(A, M, subset_by_index=[0, 0])
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector.real
    min_eigenvalue = min_eigenvalue.real
    return min_eigenvalue, min_eigenvector


###########################################################################
# Graph Partition Portion

# Generate P matrix using pymetis, making sure A fits the particular format for pymetis
def coarse_matrix(A, nc):  # A is the matrix, nc is the size of coarse matrix
    _, partition_info = pymetis.part_graph(nc, adjacency=A)
    P = []
    for i in range(nc):
        flipped_array = [1 if val == i else 0 for val in partition_info]
        if len(P) == 0:
            P = np.array(flipped_array)
        else:
            P = np.column_stack((P, np.array(flipped_array)))
    return P


# Generate correct format of adjacency matrix for the graph partition function
def generate_adjacency(graph_laplacian):
    diag_value = graph_laplacian.diagonal()
    degree_matrix = sp.sparse.diags(diag_value, format='csr')
    adjacency_matrix = degree_matrix - graph_laplacian
    G = nx.from_scipy_sparse_array(adjacency_matrix)
    adjacency_list = nx.to_dict_of_lists(G)
    return adjacency_list, degree_matrix


# Generate a list of coarse matrix A,M, and a list of P matrix
def generate_coarse_graph(nc, adj, A, M):
    if len(nc) == 0:
        return None, None, None, None

    # Store information for later-use
    coarse_matrix_storage = []
    p_info_storage = []
    coarse_diagonal_matrix_storage = []
    coarse_vector_storage = []

    np.random.seed(50)

    ac = A
    mc = M

    for i in range(len(nc)):
        # Update P matrix
        p = coarse_matrix(adj, nc[i])
        p = update_coarse_p(p, nc, i)
        p = sp.sparse.csr_matrix(p)  # make sure it is sparse format
        p_info_storage.append(p)

        ac = p.T @ (ac @ p)
        coarse_matrix_storage.append(ac)
        adj, degree_matrix = generate_adjacency(ac)  # create correct format of adjacency matrix for graph

        mc = p.T @ (mc @ p)
        coarse_diagonal_matrix_storage.append(mc)

        v = np.random.rand(ac.shape[0]+1)
        coarse_vector_storage.append(v)

    return coarse_matrix_storage, coarse_diagonal_matrix_storage, p_info_storage, coarse_vector_storage


# delete columns in P that does not have any nodes and update nc matrix
def update_coarse_p(P, nc, n):
    shape = P.shape[1]
    # calculate column sums
    column_sums = np.sum(P, axis=0)
    # find which columns are 0
    columns_to_delete = np.where(column_sums == 0)[0]
    # delete them in P
    P = np.delete(P, columns_to_delete, axis=1)
    shape -= len(columns_to_delete)
    nc[n] = shape
    return P


##############################################################

# create matrix in the coarse level
def a_coarse(Ac, v, A, P_current):
    n = Ac.shape[0]
    test = np.zeros((n + 1, n + 1))

    Av = A @v
    temp = Av
    for p in P_current:
        temp = p.T @ temp

    test[0][0] = v.T @ Av
    test[0][1:] = temp.T
    test[1:, 0] = temp
    Ac = Ac.toarray()
    test[1:, 1:] = Ac
    return test


def solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size):
    Acv = a_coarse(Ac, v, A, P_current)
    Mcv = a_coarse(Mc, v, M, P_current)

    n = len(P_current)
    if n == size:  # Direct Solve
        _, eigenvector = find_eigen(Acv, Mcv)
    else:  # Coordinate Descent
        eigenvector = update_v(vc, Acv, Mcv)

    alpha = eigenvector[0]
    beta = eigenvector[1:]
    vc = (1 / alpha) * beta

    temp, _ = find_eigen(Acv, Mcv)  ############

    for i in range(n - 1, -1, -1):
        P = P_current[i]
        vc = P @ vc

    v = v + vc

    norm = v.T @ M @ v

    norm = math.sqrt(norm)
    v = v / norm

    return v


def method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, v, A, M):
    if not P_info_storage:
        return v
    size = len(P_info_storage)

    if size == 1:
        Ac = coarse_matrix_storage[0]
        Mc = coarse_diagonal_matrix_storage[0]
        vc = coarse_vector_storage[0]
        P_current = P_info_storage
        v = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size)
    if size > 1:
        for i in range(size - 1, -1, -1):
            Ac = coarse_matrix_storage[i]
            Mc = coarse_diagonal_matrix_storage[i]
            vc = coarse_vector_storage[i]
            P_current = P_info_storage[:i + 1]
            v = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size)

    return v
##############################################################
