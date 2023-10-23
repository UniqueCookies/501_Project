import numpy as np
import pymetis
import set_up


# create 2x2 matrix
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
    B = np.dot(np.linalg.inv(M), A)
    eigenvalues, eigenvectors = np.linalg.eig(B)
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    return min_eigenvalue, min_eigenvector


# update v for each element
def update_v(v, A, M):
    for i in range(v.size):
        test_a = a_test(v, i, A)
        test_m = a_test(v, i, M)
        _, min_eigenvector = find_eigen(test_a, test_m)
        t = min_eigenvector[1] / min_eigenvector[0]
        v[i] += t
    norm = np.dot(v, np.dot(M, v))
    v = v / norm
    return v


#########################################################################
# two-level functions
# create P matrix
def coarse_matrix(A, nc):  # A is the matrix, nc is the size of coarse matrix
    _, partition_info = pymetis.part_graph(nc, adjacency=A)
    flipped_array = [1 if val == 0 else 0 for val in partition_info]
    membership = np.array(partition_info)
    P = np.column_stack((np.array(partition_info), np.array(flipped_array)))
    return P


def A_p_coarse(v, A, P):
    v = np.array(v)
    vAv = np.dot(v, np.dot(A, v))
    va1 = np.dot(v, np.dot(A, P))
    va2 = va1.T
    pAp = np.dot(P.T, np.dot(A, P))

    test_upper = np.insert(va1, 0, vAv)
    test_lower = np.column_stack((va2, pAp))
    test = np.vstack((test_upper, test_lower))

    return test


# update v for each element
def update_v_coarse(v, A, M):
    Ap = A_p_coarse(v, A, P)
    Mp = A_p_coarse(v, M, P)
    _, min_eigenvector = find_eigen(Ap, Mp)

    alpha = min_eigenvector[0]
    beta = min_eigenvector[1:]

    v = v + (1 / alpha) * np.dot(P, beta)

    norm = np.dot(v, np.dot(M, v))
    v = v / norm
    return v


def finish_v_coarse(nc, A, M):
    P = coarse_matrix(A, nc)
    v = np.array([0.5 for i in range(A.shape[0])])
    v = update_v_coarse(v, A, M)
    return v


##############################################################
# A, M = set_up.make_graph()
A, M = set_up.make_graph_2(1000)

correct_answer, smallest_eigenvector = find_eigen(A, M)
print(f"The correct eigenvalue is {correct_answer}")

'''''''''''
nc = 2
P = coarse_matrix(A, nc)
v = np.array([0.5 for i in range(A.shape[0])])
v = finish_v_coarse(nc, A, M)


'''''''''''
max_iteration = 10000
tol = 1e-7
iteration = 0
tolerance = 1000

#get new_v
nc = 2
P = coarse_matrix(A, nc)
v = np.array([1 for i in range(A.shape[0])])
v = finish_v_coarse(nc,A,M)

sigma = 100
while iteration <= max_iteration and tolerance > tol:
    v = update_v(v, A, M)
    top = np.dot(v, np.dot(A, v))
    bottom = np.dot(v, np.dot(M, v))
    sigma = top / bottom
    tolerance = abs(sigma - correct_answer)
    iteration += 1

if tolerance <= tol:
    print(f"Converged to the desired tolerance with {iteration} steps.")
else:
    print("Maximum iterations reached without achieving the desired tolerance.")

print("Final Tolerance:", tolerance)
print("Approximated Eigenvalue (sigma):", sigma)
