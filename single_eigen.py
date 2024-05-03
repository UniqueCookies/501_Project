import networkx as nx
import numpy as np
import set_up
import multi
import time
import scipy as sp


coarse_matrix_list = ['matrix_8897', 'matrix_2273', 'matrix_593', 'matrix_161','matrix_47']
# Read data from file
with open('matrix_35201', 'r') as file:
    lines = file.readlines()
A = set_up.create_matrix(lines)
M = set_up.create_diagonal(A)
P_current = []

P_info_storage = []
p_matrix_list = ['P_4', 'P_3', 'P_2', 'P_1','P_0']
for name in p_matrix_list:
    with open(name, 'r') as file:
        lines = file.readlines()
    p = set_up.create_matrix(lines)
    P_info_storage.append(p)

coarse_matrix_storage = []
coarse_diagonal_matrix_storage = []
coarse_vector_storage = []
mc = M
for i in range (len(coarse_matrix_list)):
    with open(coarse_matrix_list[i], 'r') as file:
        lines = file.readlines()
    ac = set_up.create_matrix(lines)
    p = P_info_storage[i]
    mc = p.T @ mc @ p
    coarse_matrix_storage.append(ac)
    coarse_diagonal_matrix_storage.append(mc)
    v = np.random.rand(ac.shape[0] + 1)
    coarse_vector_storage.append(v)


'''''''''
total_size = 10000
nc = []
k = (total_size//2)//2
print(f"Neigher nodes are {k},so each node is {k/total_size*100}% with other nodes")
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,30)
print(G)
'''''''''

v = np.random.rand(A.shape[0], 1)

start_time = time.time()
correct_answer, smallest_eigenvector = sp.sparse.linalg.lobpcg(A, X=v, B=M, largest=False, maxiter=10000)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for using sp.sparse.linalg.lobpcg is :", elapsed_time, "seconds")
print(f"correct_answer is : {correct_answer}")

left = A @ smallest_eigenvector
right = correct_answer * (M @ smallest_eigenvector)
print(f"The norm of error Av = \lambda Mv is: {np.linalg.norm(left - right)}")
'''''''''

start_time = time.time()
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = multi.generate_coarse_graph(
    nc, adj, A, M)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for creating necessary coarse matrix is :", elapsed_time, "seconds")

if P_info_storage is not None:
    for num in P_info_storage:
        print(num.shape)


tolerance = 1000
iteration = 0
MAXINTERATION = 100
old = 1000

start_time = time.time()
while tolerance > 1e-10 and iteration < MAXINTERATION:
    v = multi.update_v(v, A, M)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom
    print(f"After coordinate descent at the fine level at iteratin {iteration}, the eigenvalue is {sigma}")

    v = multi.method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, v, A,
                     M)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom

    if iteration % 10 == 0:
        print(f"After iteration {iteration}, the eigenvalue is {sigma}")

    tolerance = abs(sigma - old)
    old = sigma
    iteration += 1

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for finding eigenvalue using multigrid is: :", elapsed_time, "seconds")

error = abs(sigma - correct_answer) / correct_answer * 100
print(f"The predicted eigenvalue is {sigma} with iteration {iteration} and error {error}%")

left = A @ v
right = sigma * (M @ v)
error_prediced_eigenvalue = np.linalg.norm(left - right)
print(f"Difference of the norm between Av and lMv {error_prediced_eigenvalue}")
'''''''''