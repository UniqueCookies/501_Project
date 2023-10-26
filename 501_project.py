import numpy as np
import scipy as sp
import pymetis
import scipy.linalg

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
    eigenvalues, eigenvectors = sp.linalg.eig(A,M)

    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector.real
    min_eigenvalue = min_eigenvalue.real
    return min_eigenvalue, min_eigenvector

def find_eigen_numpy(A, M):
    K = np.dot(np.linalg.inv(M),A)
    eigenvalues, eigenvectors = np.linalg.eig(K)

    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index]
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector.real
    min_eigenvalue = min_eigenvalue.real
    return min_eigenvalue, min_eigenvector


# update v for each element
def update_v(v, A, M):
    for i in range(v.size):
        test_a = a_test(v, i, A)
        test_m = a_test(v, i, M)
        _, min_eigenvector = find_eigen_numpy(test_a, test_m)
        t = min_eigenvector[1] / min_eigenvector[0]
        v[i] += t
    norm = np.dot(v, np.dot(M, v))
    v = v / norm
    return v


#########################################################################
# two-level functions


def A_p_coarse(v, A, P):
    v = np.array(v)
    vAv = np.dot(v.T, np.dot(A, v))
    va1 = np.dot(v.T, np.dot(A, P))
    pAp = np.dot(P.T, np.dot(A, P))
    if isinstance(va1, (int, float)):
        test_upper = np.array([vAv,va1])
        test_lower = np.array([va1,pAp])
    else:
        test_upper = np.insert(va1, 0, vAv)
        test_lower = np.column_stack((va1.T, pAp))
    test = np.vstack((test_upper, test_lower))
    return test


# update v for each element
def update_v_coarse(v, A, M, P):
    Ap = A_p_coarse(v, A, P)
    Mp = A_p_coarse(v, M, P)

    _, min_eigenvector = find_eigen(Ap, Mp)
    alpha = min_eigenvector[0]
    beta = min_eigenvector[1:]

    v = v + (1 / alpha) * np.dot(P, beta)
    norm = np.dot(v, np.dot(M, v))
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


def finish_v_coarse_multi(nc, A, M,adj):
    P = coarse_matrix(adj, nc)
    v = np.array([1 for i in range(A.shape[0])])
    v = update_v_coarse(v, A, M, P)
    return v


def v_coarse_multi(total_size, A, M,adj,decrase_coarse):
    v = np.array([1 for i in range(A.shape[0])])
    nc = total_size//decrase_coarse
    P = coarse_matrix(adj, nc)
    v = v_coarse_recursion(nc,v,A,M,P,decrase_coarse)
    return v

def v_coarse_recursion(nc,v,A,M,P,decrase_coarse):
    if nc < decrase_coarse+1 or nc<5:
        v = update_v_coarse(v, A, M, P)
        return v
    nc = int(nc//decrase_coarse)
    P = coarse_matrix(adj, nc)
    v = v_coarse_recursion(nc,v,A,M,P,decrase_coarse)
    v = update_v_coarse(v, A, M, P)
    return v


##############################################################
# A, M = set_up.make_graph()
adj, A, M = set_up.make_graph_2(1000)
#correct_answer, smallest_eigenvector = find_eigen_numpy(A, M)
correct_answer= 0.0004810690277549212
#print(f"The correct eigenvalue is {correct_answer}")

##############################################################
decrease_coarse = 7
total_size = A.shape[0]
v= v_coarse_multi(total_size, A, M, adj, decrease_coarse)
v = v.real
#print(f"Eigenvector for coarse: {v}")
v0 = v
######
max_iteration = 1000
tol = 1e-7
iteration = 0
tolerance = 1000

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

#print("Final Tolerance:", tolerance)
#print("Approximated Eigenvalue (sigma):", sigma)
#print(f"Eigenvector: {v}")

print(f"difference is:{np.linalg.norm(v-v0)}")

##############################################################
#Two-level method
'''''''''''
max_iteration = 10000
tol = 1e-7
iteration = 0
tolerance = 1000

#get new_v
nc = 2
v = np.array([1 for i in range(A.shape[0])])
v = finish_v_coarse_multi(nc, A, M,adj)
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
'''''''''''