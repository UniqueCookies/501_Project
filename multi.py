import math
import time
import numpy as np
import scipy as sp
import pymetis
import scipy.linalg
from collections import Counter
import set_up


# Set up for the coordinate descent method
# create 2x2 matrix for
def a_test(v, i, A):
    vAv = np.dot(v, np.dot(A, v))
    va1 = np.dot(v, A[i])
    test = np.zeros((2, 2))
    test[0][0] = vAv
    test[0][1] = va1
    test[1][0] = va1
    test[1][1] = A[i][i]
    return test


# Coordinate descent method
def update_v(v, A, M):
    for i in range(v.size):
        test_a = a_test(v, i, A)
        test_m = a_test(v, i, M)
        min_eigenvalue, min_eigenvector = find_eigen(test_a, test_m)
        t = min_eigenvector[1] / min_eigenvector[0]
        v[i] += t
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v


# Setup Av and Mv for the LOPCG method
def a_bar(v, r_bar, v_old, A):
    if (np.all(v_old == 0)):
        test = np.zeros((2, 2))
    else:
        test = np.zeros((3, 3))

        # 3x3 portion
        vAv_old = np.dot(v, np.dot(A, v_old))
        test[0][2] = vAv_old
        test[2][0] = vAv_old

        rAv_old = np.dot(r_bar, np.dot(A, v_old))
        test[1][2] = rAv_old
        test[2][1] = rAv_old

        v_oldAv_old = np.dot(v_old, np.dot(A, v_old))
        test[2][2] = v_oldAv_old

    vAv = np.dot(v, np.dot(A, v))
    test[0][0] = vAv
    rAr = np.dot(r_bar, np.dot(A, r_bar))
    test[1][1] = rAr

    vAr = np.dot(v, np.dot(A, r_bar))
    test[0][1] = vAr
    test[1][0] = vAr

    # check positive definiteness
    if not check_positive(test):
        print("Error: Av or Mv is not positive definite")
    return test


# LOPCG method in replace of coordinate descent
def update_v_LOPCG(A, M, v, v_old, sigma):
    # for preconditioner: right now just identity matrix
    I = np.eye(A.shape[0])

    # create r_bar
    if sigma == 0:
        sigma = np.dot(v, np.dot(A, v)) / np.dot(v, np.dot(M, v))
    r = np.dot(A, v) - sigma * np.dot(M, v)
    r_bar = np.dot(I, r)

    # generate Av, Mv
    a_test = a_bar(v, r_bar, v_old, A)
    m_test = a_bar(v, r_bar, v_old, M)

    # find eigenvalue and eigenvector
    min_eigenvalue, min_eigenvector = find_eigen(a_test, m_test)

    sigma = min_eigenvalue
    alpha = min_eigenvector[0]
    beta = min_eigenvector[1]
    if (np.all(v_old == 0)):
        gamma = 0
    else:
        gamma = min_eigenvector[2]

    y = beta * r_bar + gamma * v_old
    v_old = v
    v = v + 1 / alpha * y

    # normalize
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm

    return v, sigma, v_old


# setup Av and Mv for multiple eigenvalues
def a_multi(A, X, W):
    p = X.shape[1]
    a_test = np.zeros((2 * p, 2 * p))

    Ax = np.dot(A, X)
    Aw = np.dot(A, W)
    xAx = np.dot(X.T, Ax)
    a_test[:p, :p] = xAx

    xAw = np.dot(X.T, Aw)
    wAx = xAw.T
    a_test[:p, p:] = xAw
    a_test[p:, :p] = wAx

    wAw = np.dot(W.T, Aw)
    a_test[p:, p:] = wAw

    return a_test


def update_v_multiple_eigenvalues(A, M, X, W):
    Av = a_multi(A, X, W)
    Mv = a_multi(M, X, W)
    check_positive(Av)
    check_positive(Mv)

    # eigenvalues, eigenvectors = sp.linalg.eigh(Av, Mv)

    # return eigenvalues,eigenvectors


###########################################################################
# find eigenvalues,vectors for the coarsest level
def find_eigen(A, M):
    eigenvalues, eigenvectors = sp.linalg.eigh(A, M)
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector.real
    min_eigenvalue = min_eigenvalue.real
    return min_eigenvalue, min_eigenvector


# check for convergence
def check_eigen(A, M, v, correct):
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    answer = top / bottom

    return abs(answer - correct)


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
    degree_matrix = np.diag(np.diag(graph_laplacian))
    adjacency_matrix = degree_matrix - graph_laplacian
    Ac = set_up.adj_to_list(adjacency_matrix)
    return Ac, degree_matrix


###########################################################################
# Checking for various features in the method
# Check if the matrix is positive definite
def check_positive(A):
    if not np.allclose(A, A.T, atol=1e-7):
        print("Matrix is not symmetric")
        return False
    eigenvalue, _ = np.linalg.eig(A)
    if np.all(eigenvalue > 0):
        # min_eigenvalue_index = np.argmin(eigenvalue)
        # print(f"Matrix is positive definite with min_eigenvalue {eigenvalue[min_eigenvalue_index]}")
        return True
    else:
        min_eigenvalue_index = np.argmin(eigenvalue)
        print(f"Matrix is not positive definite with eigenvalue {eigenvalue[min_eigenvalue_index]}")
        return False
    return False


# Check if the matrix is laplacian
def check_laplacian(Ac):
    n = Ac.shape[0]
    check = np.dot(Ac, np.ones(n))
    negaive_count = np.sum(check < 0)
    if negaive_count > 0:
        return False
    return True


# check if diagonal is positive
def check_diagonal(A):
    n = A.shape[0]
    for i in range(n):
        if A[i][i] <= 0:
            print(f"Not Positive Definite at {i} and the value is {A[i][i]}")
    return


###########################################################################
# Two-level method

# Create Av or Mv matrix to calculate eigenvalues
def a_p_coarse(v, A, P):
    v = np.array(v)
    vAv = np.dot(v.T, np.dot(A, v))
    va1 = np.dot(v.T, np.dot(A, P))
    pAp = np.dot(P.T, np.dot(A, P))

    if isinstance(va1, (int, float)):
        test_upper = np.array([vAv, va1])
        test_lower = np.array([va1, pAp])
    else:
        test_upper = np.insert(va1, 0, vAv)
        test_lower = np.column_stack((va1.T, pAp))
    coarse = np.vstack((test_upper, test_lower))

    return coarse


# update v in the coarse matrix
def update_v_coarse(v, A, M, P):
    Ap = a_p_coarse(v, A, P)
    Mp = a_p_coarse(v, M, P)

    _, min_eigenvector = find_eigen(Ap, Mp)
    alpha = min_eigenvector[0]
    beta = min_eigenvector[1:]

    v = v + (1 / alpha) * np.dot(P, beta)
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v


##############################################################
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

        #Update P matrix
        p = coarse_matrix(adj, nc[i])
        p = update_coarse_p(p, nc, i)
        p_info_storage.append(p)


        ac = np.dot(p.T, np.dot(ac,p))
        if not check_laplacian(ac):  # check if it is laplacian
            print(f"Error: Coarse matrix generated is NOT a laplacian matrix when coarse = {nc[i]}")
        coarse_matrix_storage.append(ac)
        adj, degree_matrix = generate_adjacency(ac)  # create correct format of adjacency matrix for graph


        mc = np.dot(p.T, np.dot(mc,p))
        if not check_positive(mc):
            print(f"Error: Coarse M matrix is not positive definite.")
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


# create matrix in the coarse level
def a_coarse(Ac, v, A, P_current):
    n = Ac.shape[0]
    test = np.zeros((n + 1, n + 1))

    Av = np.dot(A,v)
    temp = Av
    for p in P_current:
        temp = np.dot(p.T,temp)

    test[0][0] = np.dot(v, Av)
    test[0][1:] = temp.T
    test[1:, 0] = temp
    test[1:, 1:] = Ac
    return test


def solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current,size):
    Acv = a_coarse(Ac, v, A, P_current)
    Mcv = a_coarse(Mc, v, M, P_current)

    n = len(P_current)
    if n==size:        #Direct Solve
        _, eigenvector = find_eigen(Acv, Mcv)
        alpha = eigenvector[0]
        beta = eigenvector[1:]
        vc = (1/alpha) * beta

    else:                           #Coordinate Descent

        vc = update_v(vc, Acv, Mcv)
        alpha = vc[0]
        beta = vc[1:]
        vc = 1/alpha * beta

    top = np.dot(vc, np.dot(Ac, vc))
    bottom = np.dot(vc, np.dot(Mc, vc))
    sigma = top / bottom
    print(f"sigma from level {n} is {sigma}")

    for i in range (n-1,-1,-1):
        P = P_current[i]
        vc = np.dot(P,vc)
    v = v+ vc
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm

    return v


def method (coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage,v,A,M):
    size = len(P_info_storage)
    if size ==0:
        return
    if size ==1:
        Ac = coarse_matrix_storage[0]
        Mc = coarse_diagonal_matrix_storage[0]
        vc = coarse_vector_storage[0]
        P_current = P_info_storage
        v = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current,size)
    if size >1:
        for i in range (size-1,-1,-1):
            Ac = coarse_matrix_storage[i]
            Mc = coarse_diagonal_matrix_storage[i]
            vc = coarse_vector_storage[i]
            P_current = P_info_storage[:i+1]
            v = solve_vc_coarst(Ac, Mc, A, M, v, vc, P_current, size)


    return v
##############################################################

# A, M = set_up.make_graph()
total_size = 3000
adj, A, M = set_up.make_graph_2(total_size)
correct_answer, smallest_eigenvector = find_eigen(A, M)
print(f"correct_answer is : {correct_answer}")
#correct_answer = 0.0004810690277549212
#10000
#correct_answer = 3.272583307180702e-05

nc = [20]
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = generate_coarse_graph(
    nc, adj, A, M)

for num in P_info_storage:
    print(num.shape)

np.random.seed(5)
v = np.random.rand(A.shape[0])

tolerance = 1000
iteration = 0
MAXINTERATION = 100

while tolerance > 1e-7 and iteration < MAXINTERATION:
    v = update_v(v, A, M)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom
    print(f"sigma from fine is {sigma}")

    v= method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage,v,A,M)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom

    tolerance = abs(sigma - correct_answer)
    iteration +=1

print(f"The predicted eigenvalue is {sigma} with iteration {iteration}")

for i in range (len(nc)):
    Ac = coarse_matrix_storage[i]
    Mc = coarse_diagonal_matrix_storage[i]
    eigenvalue,_ =find_eigen(A, M)
    print(f"Eigenvalue of level {i} is actually {eigenvalue}")
