from multi_eigen import single_level_multi_eigen, generate_initial_vectors,coordinate_descent_matrix
import set_up
import numpy as np
import scipy as sp
from scipy.linalg import eigh

total_size = 100
num_of_eigen = 4

k = 3
print(f"Neigher nodes are {k},so each node is {k/total_size*100}% with other nodes")
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,30)
print(G)






X = generate_initial_vectors(total_size, num_of_eigen, M)

i = 0
W = np.zeros((A.shape[0],1))
W[i,0] = 1
A_new = coordinate_descent_matrix(X, i, A)
M_new = coordinate_descent_matrix(X, i, M)
num_of_eigenvalue = X.shape[1] #number of eigenvalue plan to solve
_, eigenvectors = eigh(A_new, M_new,subset_by_index =[0,(num_of_eigenvalue-1)])  # return normalized eigenvectors
alpha = eigenvectors[0, :]
beta = eigenvectors[1:,:]
print(beta)
X = X * alpha
print(X.shape)


