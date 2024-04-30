import networkx
import numpy as np
import set_up
import multi
import time
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt


total_size = 700
k = 2
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,10)
print(G)


np.random.seed(5)
v = np.random.rand(A.shape[0],1)

start_time = time.time()
correct_answer, smallest_eigenvector = sp.sparse.linalg.lobpcg(A,X=v,B=M,largest=False,maxiter=10000)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for using sp.sparse.linalg.lobpcg is :", elapsed_time, "seconds")
print(f"correct_answer is : {correct_answer}")

left = A @ smallest_eigenvector
right = correct_answer * (M @ smallest_eigenvector)
print(f"The norm of error Av = \lambda Mv is: {np.linalg.norm(left-right)}")


nc = [100,10]

start_time = time.time()
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = multi.generate_coarse_graph(
    nc, adj, A, M)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for creating necessary coarse matrix is :", elapsed_time, "seconds")

for num in P_info_storage:
    print(num.shape)

np.random.seed(5)
v = np.random.rand(A.shape[0])



tolerance = 1000
iteration = 0
MAXINTERATION = 1000
old = 1000

start_time = time.time()
while tolerance > 1e-8 and iteration < MAXINTERATION:
    v = multi.update_v(v, A, M)
    v= multi.method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage,v,A,M)
    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom

    tolerance = abs(sigma - old)
    old = sigma
    iteration +=1

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for finding eigenvalue using multigrid is: :", elapsed_time, "seconds")

error = abs(sigma - correct_answer)/correct_answer *100
print(f"The predicted eigenvalue is {sigma} with iteration {iteration} and error {error}%")

left = A @ v
right = sigma * M @ v
error_prediced_eigenvalue = np.linalg.norm(left - right)
print(f"Difference of the norm between Av and lMv {error_prediced_eigenvalue}")
