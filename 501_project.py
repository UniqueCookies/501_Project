import math
import time
import numpy as np
import scipy as sp
import pymetis
import scipy.linalg
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
    if not np.array_equal(A, A.T):
        print(A)
        return False
    eigenvalue, _ = np.linalg.eig(A)
    if eigenvalue[0] > 0:
        return True
    else:
        print(A)
        print("Error")
        return False
    print(A)
    return False


# Check if the matrix is laplacian
def check_laplacian(Ac):
    n = Ac.shape[0]
    check = np.dot(Ac, np.ones(n))
    negaive_count = np.sum(check < 0)
    if negaive_count > 0:
        return False
    return True


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
        return False

    # Store information for later-use
    coarse_matrix_storage = []
    P_info_storage = []
    A_p_storage = []
    coase_diagonal_matrix_storage = []
    M_p_storage = []

    Ac = A
    Ap = A
    Mc = M
    Mp = M
    for i in range(len(nc)):
        P = coarse_matrix(adj, nc[i])
        P_info_storage.append(P)
        Ac = np.dot(P.T, np.dot(Ac, P))
        if not check_laplacian(Ac):  # check if it is laplacian
            print(f"Error: Coarse matrix generated is NOT a laplacian matrix when coarse = {nc[i]}")
        coarse_matrix_storage.append(Ac)
        adj, degree_matrix = generate_adjacency(Ac)  # create correct format of adjacency matrix for graph

        Mc = np.dot(P.T, np.dot(Mc, P))
        if not check_positive(Mc):
            print(f"Error: Coarse M matrix is not positive definite.")
        coase_diagonal_matrix_storage.append(Mc)

        Ap = np.dot(Ap, P)
        A_p_storage.append(Ap)
        Mp = np.dot(Mp, P)
        M_p_storage.append(Mp)

    return coarse_matrix_storage, coase_diagonal_matrix_storage, A_p_storage, M_p_storage, P_info_storage


# generate coarse matrix for each coarse level except the last one
def generate_av_coarse_multi(Ap, Ac, A, v):
    v = np.array(v)
    vAv = np.dot(v.T, np.dot(A, v))
    va1 = np.dot(v.T, Ap)
    pAp = Ac

    if isinstance(va1, (int, float)):
        test_upper = np.array([vAv, va1])
        test_lower = np.array([va1, pAp])
    else:
        test_upper = np.insert(va1, 0, vAv)
        test_lower = np.column_stack((va1.T, pAp))
    test = np.vstack((test_upper, test_lower))

    return test


def coordinate_descent_av(Ap, i):
    test = np.zeros((2, 2))
    test[0][0] = Ap[0][0]
    test[0][1] = Ap[0][i+1]
    test[1][0] = Ap[i+1][0]
    test[1][1] = Ap[i+1][i+1]
    return test


def coordinate_descent_coarse(Ap,Mp,v,M):
    size = v.size
    for i in range(size-1):
        test_a = coordinate_descent_av(Ap, i)
        test_m = coordinate_descent_av(Mp, i)

        min_eigenvalue, min_eigenvector = find_eigen(test_a, test_m)
        t = min_eigenvector[1]
        v[i] += t
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v


########################################################
def update_v_coarse_multi(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage, A_p_storage, M_p_storage,
                          P_info_storage, nc):
    n = len(nc) - 1

    # start with the coarsest level
    Ap = generate_av_coarse_multi(A_p_storage[n], coarse_matrix_storage[n], A, v)
    Mp = generate_av_coarse_multi(M_p_storage[n], coarse_diagonal_matrix_storage[n], M, v)

    min_eigenvalue, min_eigenvector = find_eigen(Ap, Mp)
    alpha = min_eigenvector[0]
    beta = min_eigenvector[1:]

    P = P_info_storage[n]
    beta = np.dot(P, beta)

    for i in range(n-1, -1, -1):
        Ap = generate_av_coarse_multi(A_p_storage[i], coarse_matrix_storage[i], A, v)
        Mp = generate_av_coarse_multi(M_p_storage[i], coarse_diagonal_matrix_storage[i], M, v)
        beta = coordinate_descent_coarse(Ap,Mp,beta,coarse_diagonal_matrix_storage[i])
        P = P_info_storage[i]
        beta = np.dot(P, beta)

    v = v + (1 / alpha) * beta
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v

def check_diagonal(A):
    n = A.shape[0]
    for i in range (n):
        if A[i][i] <=0:
            print(f"Not Positive Definite at {i} and the value is {A[i][i]}")
    return
##############################################################
# A, M = set_up.make_graph()
total_size = 1000
adj, A, M = set_up.make_graph_2(total_size)
# correct_answer, smallest_eigenvector = find_eigen_numpy(A, M)
correct_answer = 0.0004810690277549212

nc = [200,40,2]
coarse_matrix_storage, coarse_diagonal_matrix_storage, A_p_storage, M_p_storage, P_info_storage = generate_coarse_graph(
 nc, adj, A, M)

for num in coarse_diagonal_matrix_storage:
    check_diagonal(num)

#generate random initial vector v
np.random.seed(50)
v = np.random.rand(A.shape[0])


tolerance = 1000
iteration = 0
MAXINTERATION = 5

while tolerance > 1e-7 and iteration<MAXINTERATION:
    v = update_v(v, A, M)
    v = update_v_coarse_multi(v, A, M, coarse_matrix_storage, coarse_diagonal_matrix_storage, A_p_storage, M_p_storage,
                          P_info_storage, nc)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom
    tolerance = abs(sigma - correct_answer)
    print(tolerance)
    iteration += 1




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
    # display_result(tolerance, tol, iteration, sigma)
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
