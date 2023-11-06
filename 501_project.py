import math
import time
import numpy as np
import scipy as sp
import pymetis
import scipy.linalg
import set_up


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


# find eigenvalues,vectors
def find_eigen(A, M):
    eigenvalues, eigenvectors = sp.linalg.eigh(A, M)
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector.real
    min_eigenvalue = min_eigenvalue.real
    return min_eigenvalue, min_eigenvector


# Check if the matrix is positive definite
def check_positive(A):
    if A != A.T:
        print("Matrix is not symmetric")
        return
    eigenvalue, _ = np.linalg.eig(A)
    if eigenvalue[0] > 0:
        print("Matrix is positive definite.")
    else:
        print("Matrix is not positive (Hermitian).")


# update v for each element in the general method
def update_v(v, A, M):
    for i in range(v.size):
        # for i in range(4):
        test_a = a_test(v, i, A)
        test_m = a_test(v, i, M)
        min_eigenvalue, min_eigenvector = find_eigen(test_a, test_m)
        # min_eigenvalue, min_eigenvector = find_eigen_numpy(test_a, test_m)
        t = min_eigenvector[1] / min_eigenvector[0]
        v[i] += t
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v


#########################################################################
# two-level functions


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
    test = np.vstack((test_upper, test_lower))

    return test


# update v for each element
def update_v_coarse(v, A, M, P):
    Ap = a_p_coarse(v, A, P)
    Mp = a_p_coarse(v, M, P)

    min_eigenvalue, min_eigenvector = find_eigen(Ap, Mp)
    alpha = min_eigenvector[0]
    beta = min_eigenvector[1:]

    v = v + (1 / alpha) * np.dot(P, beta)
    norm = np.dot(v, np.dot(M, v))
    norm = math.sqrt(norm)
    v = v / norm
    return v


##############################################################
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


def finish_v_coarse_double(nc, A, M, adj, v):
    P = coarse_matrix(adj, nc)
    v = update_v_coarse(v, A, M, P)
    return v


def v_coarse_multi(total_size, A, M, adj, decrease_coarse, v):
    nc = total_size // decrease_coarse
    P = coarse_matrix(adj, nc)
    v = v_coarse_recursion(nc, v, A, M, P, decrease_coarse)
    return v


def v_coarse_recursion(nc, v, A, M, P, decrease_coarse):
    if nc < decrease_coarse + 1 or nc < 5:
        v = update_v_coarse(v, A, M, P)
        return v
    nc = int(nc // decrease_coarse)
    P = coarse_matrix(adj, nc)
    v = v_coarse_recursion(nc, v, A, M, P, decrease_coarse)
    v = update_v_coarse(v, A, M, P)
    return v


def v_coarse_multi2(total_size, nc, A, M, adj, v):
    if nc is None:
        return False
    nc = [total_size // i for i in nc]
    for size_of_partition in nc:
        P = coarse_matrix(adj, size_of_partition)
        v = update_v_coarse(v, A, M, P)
    return v


def generate_coarse_graph_wrapper(nc):
    if len(nc) == 0:
        return False
    P = coarse_matrix(adj, nc[2])
    Ac = generate_coarse_graph(nc, 2, P)
    return Ac


def generate_coarse_graph(nc, n, P):
    Ac = np.dot(P.T, np.dot(A, P))
    if not check_laplacian(Ac):  # check if it is laplacian
        print("Error: Not a laplacian matrix")
    #Ac = set_up.adj_to_list(Ac) # change to the right format
    return Ac


def check_laplacian(Ac):
    n = Ac.shape[0]
    check = np.dot(Ac, np.ones(n))
    negaive_count = np.sum(check < 0)
    if negaive_count > 0:
        return False
    zero_count = np.sum(check == 0)
    if zero_count / len(check) > 0.9:
        return True
    else:
        return False


##############################################################
# A, M = set_up.make_graph()
total_size = 1000
adj, A, M = set_up.make_graph_2(total_size)
# correct_answer, smallest_eigenvector = find_eigen_numpy(A, M)
correct_answer = 0.0004810690277549212

nc = [500, 200, 100]
nc = generate_coarse_graph_wrapper(nc)
print(nc)


##############################################################
# Two-level method
def main_method(A, M, nc):
    np.random.seed(50)
    v = np.random.rand(A.shape[0])

    # Loop Set up with Tol
    sigma = 1000
    max_iteration = 100
    tol = 1e-7
    iteration = 0
    tolerance = 1000
    start_time = time.process_time()
    while iteration <= max_iteration and tolerance > tol:
        # fine level
        v = update_v(v, A, M)

        # coarse level
        # v = finish_v_coarse_double(nc, A, M, adj, v)  # Two-level method
        # v = v_coarse_multi(1000, A, M, adj, nc, v)          # Multi-level method
        v = v_coarse_multi2(total_size, nc, A, M, adj, v)
        # calculate eigenvalue
        sigma_old = sigma
        top = np.dot(v, np.dot(A, v))
        bottom = np.dot(v, np.dot(M, v))
        sigma = top / bottom

        # Compare with real value
        # tolerance = abs(sigma - correct_answer)
        tolerance = abs(sigma - sigma_old)
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


'''''''''
# Tow-level Method
#nc = [2**i for i in range(1,10)]
for partition in nc:
    iteration,sigma,actual_error,process_time= main_method(A, M, partition)
    print(f"Iteration: {iteration} with coarse: {partition}. Actual Error: {actual_error}. Time used: {process_time}")
'''''''''
'''''''''
nc = np.array([500,200,100])
iteration,sigma,actual_error,time_used = main_method(A, M, nc)
print(f"Iteration: {iteration} with coarse: {nc}. Actual Error: {actual_error}. Time used: {time_used}")
'''''''''
