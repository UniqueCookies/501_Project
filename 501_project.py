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
    P_info_storage = []
    coase_diagonal_matrix_storage = []
    coarse_vector_storage = []
    A_p_storage = []
    M_p_storage = []

    np.random.seed(50)

    Ac = A
    Mc = M
    A_p = A
    M_p = M
    for i in range(len(nc)):
        P = coarse_matrix(adj, nc[i])

        P = update_coarse_p(P, nc, i)

        P_info_storage.append(P)

        A_p = np.dot(A_p, P)
        Ac = np.dot(P.T, np.dot(Ac, P))
        if not check_laplacian(Ac):  # check if it is laplacian
            print(f"Error: Coarse matrix generated is NOT a laplacian matrix when coarse = {nc[i]}")
        A_p_storage.append(A_p)
        coarse_matrix_storage.append(Ac)
        adj, degree_matrix = generate_adjacency(Ac)  # create correct format of adjacency matrix for graph

        M_p = np.dot(M_p, P)
        Mc = np.dot(P.T, np.dot(Mc, P))
        if not check_positive(Mc):
            print(f"Error: Coarse M matrix is not positive definite.")
        coase_diagonal_matrix_storage.append(Mc)
        M_p_storage.append(M_p)

        v = np.random.rand(Ac.shape[0])
        coarse_vector_storage.append(v)

    return coarse_matrix_storage, coase_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, A_p_storage, M_p_storage


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


# generate coarse matrix Av to calculate the weights from coarse and current level
def a_test_multi(Ac, vc, Pvc):
    test = np.zeros((2, 2))

    vAv = np.dot(vc.T, np.dot(Ac, vc))
    test[0][0] = vAv
    va1 = np.dot(vc.T, np.dot(Ac, Pvc))
    test[0][1] = va1
    test[1][0] = va1

    pAp = np.dot(Pvc.T, np.dot(Ac, Pvc))
    test[1][1] = pAp

    return test


# calculate weights and combining info from the coarse level with the current level
def update_v_multi_initial(Ac, Mc, v, Pvc):
    Av = a_test_multi(Ac, v, Pvc)
    Mv = a_test_multi(Mc, v, Pvc)
    eigenvalue, eigenvector = find_eigen(Av, Mv)
    # print(f"weight is {eigenvector}")
    return eigenvector[0] * v + eigenvector[1] * Pvc


# returning back to fine level in multi-level
def update_v_upward(Ac, Mc, P, vc, v):
    # initial guess for v
    Pvc = np.dot(P, vc)
    v = update_v_multi_initial(Ac, Mc, v, Pvc)

    # gradient descent
    v = update_v(v, Ac, Mc)

    return v


# create matrix in the coarsest level --> may not be the correct algorithm
def a_coarsest(Ac, vc, Ap, A):
    n = Ac.shape[0]
    test = np.zeros((n + 1, n + 1))
    temp = np.dot(vc.T, Ap)
    test[0][0] = np.dot(vc, np.dot(A, vc))
    test[0][1:] = temp
    test[1:, 0] = temp
    test[1:, 1:] = Ac
    return test


def solve_vc_coarst(Ac, Mc, Ap, Mp, A, M, P, vc):
    Acv = a_coarsest(Ac, vc, Ap, A)
    Mcv = a_coarsest(Mc, vc, Mp, M)
    check_positive(Acv)
    check_positive(Mcv)
    _, eigenvector = find_eigen(Acv, Mcv)
    alpha = eigenvector[0]
    beta = eigenvector[1:]

    vc = vc + 1 / alpha * np.dot(P, beta)

    norm = np.dot(vc, np.dot(M, vc))
    norm = math.sqrt(norm)
    vc = vc / norm

    return vc


########################################################
def update_v_coarse_multi(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage,
                          P_info_storage, coarse_vector_storage, nc):
    n = len(nc)
    # fine level
    v = update_v(v, A, M)

    # gradient descent downward
    for i in range(n - 2):
        Ac = coarse_matrix_storage[i]
        Mc = coarse_diagonal_matrix_storage[i]
        coarse_vector_storage[i] = update_v(coarse_vector_storage[i], Ac, Mc)

    # at the finest level
    _, coarse_vector_storage[n - 1] = find_eigen(coarse_matrix_storage[n - 1], coarse_diagonal_matrix_storage[n - 1])

    # continue backward
    for i in range(n - 2, 0, -1):
        Ac = coarse_matrix_storage[i]
        Mc = coarse_diagonal_matrix_storage[i]
        P = P_info_storage[i + 1]
        vc = coarse_vector_storage[i + 1]
        v = coarse_vector_storage[i]
        coarse_vector_storage[i] = update_v_upward(Ac, Mc, P, vc, v)

    return np.dot(P_info_storage[0], coarse_vector_storage[0])


def update_v_coarse_multi2(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage,
                           P_info_storage, coarse_vector_storage, A_p_storage, M_p_storage, nc):
    # fine level
    v = update_v(v, A, M)

    n = len(nc)

    # Case 1: Single Level - No Coarse
    if n == 0:
        return v

    # Case 2: Tow-Level Method
    if n == 1:
        print("two level method ")
        Ac = coarse_matrix_storage[0]
        Mc = coarse_diagonal_matrix_storage[0]
        Ap = A_p_storage[0]
        Mp = M_p_storage[0]
        P = P_info_storage[0]

        vc = v
        # Method: directly solving with coarse information
        # _, vc = find_eigen(Ac, Mc)  #directly solving in the coarsest level
        # vc = coarse_vector_storage[0] #use its own coarse vector -> cause the matrix to have 0 eigenvalue
        # vc = np.dot (v,P)
        vc = solve_vc_coarst(Ac, Mc, Ap, Mp, A, M, P, vc)  # incorporate vc information
        # vc = update_v(vc, Ac, Mc) #gradient descent on the coarsest -> does not help with result that much
        coarse_vector_storage[0] = vc  # update coarse

        # Convergence checking for the coarse level
        # top = np.dot(v, np.dot(A, v))
        # bottom = np.dot(v, np.dot(M, v))
        # sigma = top / bottom
        # tolerance = abs(sigma - 0.0004810690277549212)
        # print(f"tolerance before coarsing is {tolerance}")

        v = update_v_multi_initial(A, M, v, vc)

        return v

    # Multilevel Method
    # gradient descent going down
    print ("multiple-level method")



    return v


##########################################################
# gram_schmidt to generate a set of orthonormal vectors
def gram_schmidt(A, M):
    (n, m) = A.shape
    W = np.zeros((n, m))

    for i in range(m):

        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        # normalize q
        q = q / np.sqrt(np.dot(q, np.dot(M, q)))

        # write the vector back in the matrix
        W[:, i] = q
    return W


# generate linearly independent initial vectors
def generate_initial_vectors(n, p):
    V = np.zeros((n, p))
    for i in range(p):
        np.random.seed(i)
        V[:, i] = np.random.rand(A.shape[0])

        iteration = 0
        MAXINTERATION = 10
        while not np.linalg.matrix_rank(V) == i + 1 and iteration < MAXINTERATION:
            np.random.seed(i + 1)
            V[:, i] = np.random.rand(A.shape[0])
            iteration += 1
    if np.linalg.matrix_rank(V) == p:
        return V
    else:
        print("Error: Unable to generate linearly independent vectors")
        return False


# double check if W is M-orthornormal
def check_if_orthonormal(W, M):
    for i in range(W.shape[1]):
        v = W[:, i]

        # check if normalized
        norm = np.dot(v, np.dot(M, v))
        if not abs(1 - norm) < 1e-5:
            print("Each vector is not M - normalized")
    return np.linalg.matrix_rank(W) == W.shape[1]


##############################################################
# A, M = set_up.make_graph()
total_size = 1000
adj, A, M = set_up.make_graph_2(total_size)
# correct_answer, smallest_eigenvector = find_eigen(A, M)
# print(correct_answer)
correct_answer = 0.0004810690277549212

nc = [200, 2]
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, A_p_storage, M_p_storage = generate_coarse_graph(
    nc, adj, A, M)

# generate random initial vector v
np.random.seed(50)
v = np.random.rand(A.shape[0])

# Set up for the LOPCG
# v_old = np.zeros(A.shape[0])
# sigma = 0

tolerance = 1000
iteration = 0
MAXINTERATION = 10

while tolerance > 1e-7 and iteration < MAXINTERATION:
    # v, sigma, v_old = update_v_LOPCG(A, M, v, v_old, sigma)
    # v= update_v(v, A, M)
    v = update_v_coarse_multi2(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage,
                               P_info_storage, coarse_vector_storage, A_p_storage, M_p_storage, nc)
    '''''''''
    if iteration < 5:
        top = np.dot(v, np.dot(A, v))
        bottom = np.dot(v, np.dot(M, v))
        sigma = top / bottom
        tolerance = abs(sigma - correct_answer)
        print(f"after first step is {tolerance}")
     '''''''''

    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom
    tolerance = abs(sigma - correct_answer)
    #print(tolerance)
    iteration += 1

print(f"Final Eigenvalue is{sigma} with iteration {iteration} and tolerance {tolerance}")


##############################################################
# Two-level method
def main_method(A, M, nc):
    np.random.seed(50)
    v = np.random.rand(A.shape[0])
    coarse_matrix_storage, coarse_diagonal_matrix_storage, A_p_storage, M_p_storage, P_info_storage = generate_coarse_graph(
        nc, adj, A, M)

    # Loop Set up with Tol
    sigma = 1000
    max_iteration = 1
    tol = 1e-7
    iteration = 0
    tolerance = 1000
    v = update_v(v, A, M)
    start_time = time.process_time()
    while iteration <= max_iteration and tolerance > tol:
        # coarse level
        # v = finish_v_coarse_double(nc, A, M, adj, v)  # Two-level method
        v = update_v_coarse_multi(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage, A_p_storage,
                                  M_p_storage, P_info_storage, nc)
        # fine level
        v = update_v(v, A, M)
        # calculate eigenvalue
        sigma_old = sigma
        top = np.dot(v, np.dot(A, v))
        bottom = np.dot(v, np.dot(M, v))
        sigma = top / bottom

        # Compare with real value
        tolerance = abs(sigma - correct_answer)
        # tolerance = abs(sigma - sigma_old)
        # print(tolerance)
        iteration += 1
    display_result(tolerance, tol, iteration, sigma)
    actual_error = abs(sigma - correct_answer)
    end_time = time.process_time()
    return iteration, sigma, actual_error, end_time - start_time


def display_result(tolerance, tol, iteration, sigma):
    # display results
    if tolerance <= tol:
        print(f"Converged to the desired tolerance with {iteration} steps.")
    else:
        print("Maximum iterations reached without achieving the desired tolerance.")

    print("Final Tolerance:", tolerance)
    print("Approximated Eigenvalue (sigma):", sigma)


# iteration, sigma, actual_error, time_used = main_method(A, M, nc)
# print(f"Iteration: {iteration} with coarse: {nc}. Actual Error: {actual_error}. Time used: {time_used}")
'''''''''


# Tow-level Method
nc = [2**i for i in range(1,10)]
for partition in nc:
    iteration,sigma,actual_error,process_time= main_method(A, M, partition)
    print(f"Iteration: {iteration} with coarse: {partition}. Actual Error: {actual_error}. Time used: {process_time}")
'''''''''
'''''''''
nc = np.array([500,200,100])
iteration,sigma,actual_error,time_used = main_method(A, M, nc)
print(f"Iteration: {iteration} with coarse: {nc}. Actual Error: {actual_error}. Time used: {time_used}")
'''''''''
