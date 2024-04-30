import numpy as np
import scipy as sp

import set_up
from multi import generate_coarse_graph
from multi_eigen import basic_calculation

total_size = 100
p = 5
G, adj, A, M = set_up.make_graph_2(total_size)

# Actual Answer
eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(A,k=2*p,M = M,which = 'SM')
v = np.sort(eigenvalues)
print(v[:2*p])


nc = [20]
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = generate_coarse_graph(
    nc, adj, A, M)

for num in P_info_storage:
    print(num.shape)

p_matrix = P_info_storage[0]
coarse_a = coarse_matrix_storage[0]
coarse_m = coarse_diagonal_matrix_storage [0]
basic_calculation(total_size, p, A, M, p_matrix,coarse_a,coarse_m)