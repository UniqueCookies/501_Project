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
    else:                           #Coordinate Descent
        eigenvector = update_v(vc, Acv, Mcv)

    alpha = eigenvector[0]
    beta = eigenvector[1:]
    vc = (1 / alpha) * beta


    temp, _ = find_eigen(Acv, Mcv) ############

    t1 = eigenvector


    top = np.dot(t1, np.dot(Acv, t1))
    bottom = np.dot(t1, np.dot(Mcv, t1))
    sigma = top / bottom

    error = abs(temp-sigma)/sigma *100

    print(f"sigma from level {n} is {sigma} but should be {temp} with error {error}%")

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
total_size = 5000
adj, A, M = set_up.make_graph_2(total_size)
correct_answer, smallest_eigenvector = find_eigen(A, M)
print(f"correct_answer is : {correct_answer}")
#correct_answer = 0.0004810690277549212
#10000
#correct_answer = 3.272583307180702e-05

nc = [500,50,5]
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = generate_coarse_graph(
    nc, adj, A, M)

for num in P_info_storage:
    print(num.shape)

np.random.seed(5)
v = np.random.rand(A.shape[0])

tolerance = 1000
iteration = 0
MAXINTERATION = 1000
old = 1000

while tolerance > 1e-8 and iteration < MAXINTERATION:
    v = update_v(v, A, M)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma_fine = top / bottom
    print(f"sigma from fine is {sigma_fine}")

    difference = abs(sigma_fine - old) / old * 100
    print(f"Decrease in eigenvalue by fine level is {difference}%")

    v= method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage,v,A,M)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom

    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom
    print(f"sigma after coarse update is {sigma}")

    difference = (sigma_fine-sigma)/sigma_fine * 100
    print(f"Decrease in eigenvalue by coarse level is {difference}%")

    tolerance = abs(sigma - old)
    old = sigma
    iteration +=1
error = abs(sigma - correct_answer)/correct_answer *100
print(f"The predicted eigenvalue is {sigma} with iteration {iteration} and error {error}%")

left = np.dot(A,v)
right = sigma * np.dot (M,v)
error_prediced_eigenvalue = np.linalg.norm(left - right)
print(f"Difference of the norm between Av and lMv{error_prediced_eigenvalue}")
