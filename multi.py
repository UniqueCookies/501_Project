import math
import networkx as nx
import numpy as np
import scipy as sp
import pymetis

import set_up


# Set up for the coordinate descent method
# create 2x2 matrix for each column of A and M
def a_test(v, i, A):
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
    adjacency_list = set_up.adj_to_list(adjacency_matrix)
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
        ac = sp.sparse.csr_matrix(ac)
        coarse_matrix_storage.append(ac)
        adj, degree_matrix = generate_adjacency(ac)  # create correct format of adjacency matrix for graph

        mc = p.T @ (mc @ p)
        mc = sp.sparse.csr_matrix(mc)
        coarse_diagonal_matrix_storage.append(mc)

        v = np.random.rand(ac.shape[0] + 1)
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

    Av = A @ v
    temp = Av
    for p in P_current:
        temp = p.T @ temp

    test[0][0] = v.T @ Av
    test[0,1:] = temp.T
    test[1:,0] = test[0,1:].T
    Ac = Ac.toarray()
    test[1:,1:] = Ac


    inverse = np.linalg.inv(Ac)
    right = test[0,1:] @ (inverse @ test[1:,0])
    return test


def update_vc_coarse_level_new(Ac, v, vc, A, P_current, i):
    upper = v.T @ A @ v
    right = v.T @ A
    for p in P_current:
        right = right @ p
    vAv = vc[0] ** 2 * upper
    temp = np.dot(right,vc[1:])
    v0rightv = vc[0] * temp
    vAcv = vc[1:].T @ Ac @ vc[1:]
    top_left = vAv + 2 * v0rightv + vAcv
    Acv_i = np.zeros((2, 2))
    Acv_i[0, 0] = top_left

    if i == 0:
        Acv_i[0, 1] = temp + vc[0] * upper
        Acv_i[1, 0] = Acv_i[0, 1]
        Acv_i[1, 1] = upper
    else:
        Acv_i[1, 1] = Ac[i - 1, i - 1]
        col = Ac.getcol(i - 1)
        left = right[i - 1] * vc[0]
        right = vc[1:].T @ col
        Acv_i[0, 1] = left + right
        Acv_i[1, 0] = Acv_i[0, 1]

    return Acv_i

def eigen_check(Ac, v, vc, A, P_current):
    upper = v.T @ A @ v
    right = v.T @ A
    for p in P_current:
        right = right @ p

    vAv = vc[0] ** 2 * upper
    temp = np.dot(right, vc[1:])
    v0rightv = vc[0] * temp
    vAcv = vc[1:].T @ Ac @ vc[1:]
    top_left = vAv + 2 * v0rightv + vAcv
    return top_left

def update_v_coarse(Ac, Mc, v, vc, A, M, P_current):
    for i in range(vc.size):
        test_a = update_vc_coarse_level_new(Ac, v, vc, A, P_current, i)
        test_m = update_vc_coarse_level_new(Mc, v, vc, M, P_current, i)
        _, min_eigenvector = sp.linalg.eigh(test_a, test_m, subset_by_index=[0, 0])

        t = min_eigenvector[1] / min_eigenvector[0]
        vc[i] += t

    top = eigen_check(Ac, v, vc, A, P_current)
    bottom = eigen_check(Mc, v, vc, M, P_current)
    sigma = top/bottom
    print(f"The eigenvalue in coarse level {len(P_current)} is converging to {sigma}")
    return vc


def solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size):
    n = len(P_current)
    if n == size:  # Direct Solve
        Acv = a_coarse(Ac, v, A, P_current)
        Mcv = a_coarse(Mc, v, M, P_current)

        _, eigenvector = find_eigen(Acv, Mcv)
    else:  # Coordinate Descent
        eigenvector = update_v_coarse(Ac, Mc, v, vc, A, M, P_current)

    alpha = eigenvector[0]
    beta = eigenvector[1:]
    vc = (1 / alpha) * beta

    for i in range(n - 1, -1, -1):
        P = P_current[i]
        vc = P @ vc

    v = v.reshape(v.size)
    v = v + vc
    norm = v.T @ M @ v

    norm = math.sqrt(norm)
    v = v / norm

    return v,eigenvector


def method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, v, A, M):
    if not P_info_storage:
        return v
    size = len(P_info_storage)

    if size == 1:
        Ac = coarse_matrix_storage[0]
        Mc = coarse_diagonal_matrix_storage[0]
        vc = coarse_vector_storage[0]
        P_current = P_info_storage
        v, vc = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size)
        coarse_vector_storage[0] = vc

    if size > 1:
        for i in range(size - 1, -1, -1):
            Ac = coarse_matrix_storage[i]
            Mc = coarse_diagonal_matrix_storage[i]
            vc = coarse_vector_storage[i]
            P_current = P_info_storage[:i + 1]
            v,vc = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size)
            coarse_vector_storage[i]=vc

    return v
##############################################################
