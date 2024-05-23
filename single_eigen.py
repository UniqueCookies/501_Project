import networkx as nx
import numpy as np
import set_up
import multi
import time
import scipy as sp
import math
import matplotlib.pyplot as plt
import pandas as pd

MAXINTERATION = 300
epsilon = 1e-6

'''''''''
facebook = pd.read_csv('facebook_combined.txt',
                      sep = " ",
                      names = ["start_node", "end_node"],
                      )
G = nx.from_pandas_edgelist(facebook,"start_node","end_node")
print(G)

pos = nx.spring_layout(G,iterations=15,seed=1721)
fig,ax = plt.subplots(figsize=(15,9))
ax.axis("off")
nx.draw_networkx(G,pos = pos, ax= ax, node_size = 10 ,with_labels=False)
plt.show()
'''''''''


# Import graph
with open('facebook_combined.txt', 'r') as file:
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    n = len(lines)
    last_line = lines[-1]
    numbers_on_last_line = [int(num) for num in last_line.strip().split()]
    m = max(numbers_on_last_line)

K = sp.sparse.lil_matrix((n, m+1))
iteration = 0
for line in lines:
    numbers = [int(num) for num in line.strip().split()]
    a = numbers[0]
    b = numbers[1]
    K[iteration,a] = 1
    K[iteration,b] = 1
    iteration +=1

A = K.T @ K
# Set Last Colume/Row to be 0
n = A.shape[0]
for i in range(n - 1):
    A[n - 1, i] = 0
    A[i, n - 1] = 0
# Eliminate unncessary zeros
A.eliminate_zeros()
A = A.tocsr()
diag = A.diagonal()
M = sp.sparse.csr_matrix((diag, [np.arange(len(diag)), np.arange(len(diag))]), shape=(A.shape[0], A.shape[1]))
A = -A+2*M
adj = set_up.adj_to_list(A)
nc = [1010,251, 62]

##################################################
'''''''''''
#Graph for result 2
A_display = A
diag = A.diagonal()
D = sp.sparse.diags(diag)
A_display = D- A
G = nx.from_scipy_sparse_array(A_display)
print(G)

pos = nx.spring_layout(G,iterations=15,seed=1721)
fig,ax = plt.subplots(figsize=(15,9))
ax.axis("off")
nx.draw_networkx(G,pos = pos, ax= ax, node_size = 10 ,with_labels=False)
plt.show()

coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = multi.generate_coarse_graph(
    nc, adj, A, M,1)

for coarse_A in coarse_matrix_storage:
    diag = coarse_A.diagonal()
    D = sp.sparse.diags(diag)
    A_display = D - coarse_A
    G = nx.from_scipy_sparse_array(A_display)
    print(f"The coarse matrix {coarse_A.shape} has the graph of {G}")

    pos = nx.spring_layout(G, iterations=15, seed=1721)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=10, with_labels=False)
    plt.show()
'''''''''''

##################################################
# Code for result 1
'''''''''''
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
'''''''''''

#####################################################################
#Graph result 1
'''''''''''
coordinates = []
with open('coord_35201', 'r') as file:
    for line in file:
        # Assuming x-y coordinates are separated by spaces or commas
        x, y = map(float, line.strip().replace(',', ' ').split())
        coordinates.append((x, y))


for i in range (len(coarse_matrix_storage)):
    Ac = coarse_matrix_storage[i]
    n = Ac.shape[0]
    temp = coordinates[:n]
    nodes = {}
    for i, coord in enumerate(temp):
        nodes[f'Node_{i+1}'] = coord

    edge_list = []
    Mc = set_up.create_diagonal(Ac)
    adj = Mc-Ac
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        # Get the indices of non-zero elements in the i-th row
        connected_nodes = adj[i].nonzero()[1]
        for node_idx in connected_nodes:
            edge_list.append((f'Node_{i+1}', f'Node_{node_idx+1}'))

    G = nx.Graph()
    G.add_edges_from(edge_list)
    G.add_nodes_from(nodes)
    nx.draw(G,pos=nodes,node_size = 10)
    plt.show()
'''''''''''
###################################################################

'''''''''''
# Create Random Graph using networkx

total_size = 500
nc = [100, 10]
k = 10
print(f"Neigher nodes are {k},so each node is {k/total_size*100}% with other nodes")
G, adj, A, M = set_up.make_graph(total_size,k,0.2,10,30)
print(G)

'''''''''''


v = np.random.rand(A.shape[0],1)
'''''''''''
# Using LOBPCG to Solve for eigenvalues

start_time = time.time()
correct_answer, smallest_eigenvector = sp.sparse.linalg.lobpcg(A, X=v, B=M, tol=1e-10, largest=False, maxiter=20000)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for using sp.sparse.linalg.lobpcg is :", elapsed_time, "seconds")

left = A @ smallest_eigenvector
right = correct_answer * (M @ smallest_eigenvector)
print(f"Correct_answer by lobpcg method is : {correct_answer} with the norm of the residual error : {np.linalg.norm(left - right)}")

'''''''''''
start_time = time.time()
coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage = multi.generate_coarse_graph(
    nc, adj, A, M,1)


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for creating necessary coarse matrix is :", elapsed_time, "seconds")

if P_info_storage is not None:
    for num in P_info_storage:
        print(num.shape)


'''''''''''


tolerance = 10
iteration = 0

# Create Starting previous value for tolerance
top = v.T @ A @ v
bottom = v.T @ M @ v
sigma = top / bottom
previous = sigma

start_time = time.time()
while tolerance > epsilon * previous and (iteration < MAXINTERATION or iteration<5):

    v = multi.update_v(v, A, M)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom
    print(f"After coordinate descent at the fine level at iteratin {iteration}, the eigenvalue is {sigma}")


    v = multi.method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, v, A,
                    M,0)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom


    left = A @ v
    right = sigma * (M @ v)
    residual_error = np.linalg.norm(left - right)
    tolerance = abs(previous-sigma)
    previous = sigma
    print(f"Current iteration {iteration+1} has predicted eigenvalue of {sigma} and residual error of {residual_error} with tolerance {tolerance}"
          f"and the norm of calculated eigenvector is {np.linalg.norm(v)}")

    iteration += 1

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for finding eigenvalue using multigrid is: :", elapsed_time, "seconds")

#error = abs(sigma - correct_answer) / correct_answer * 100
#print(f"The predicted eigenvalue is {sigma} with iteration {iteration} and error {error}%")

left = A @ v
right = sigma * (M @ v)
residual_error = np.linalg.norm(left - right)
print(f"The residual error is {residual_error}")

##########################################################################
# Only the Coarses:

v = np.random.rand(A.shape[0],1)
tolerance = 10
iteration = 0

# Create Starting previous value for tolerance
top = v.T @ A @ v
bottom = v.T @ M @ v
sigma = top / bottom
previous = sigma

start_time = time.time()
while tolerance > epsilon * previous and (iteration < MAXINTERATION or iteration<5):

    v = multi.update_v(v, A, M)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom
    print(f"After coordinate descent at the fine level at iteratin {iteration}, the eigenvalue is {sigma}")


    v = multi.method(coarse_matrix_storage, coarse_diagonal_matrix_storage, P_info_storage, coarse_vector_storage, v, A,
                    M,1)

    top = v.T @ A @ v
    bottom = v.T @ M @ v
    sigma = top / bottom


    left = A @ v
    right = sigma * (M @ v)
    residual_error = np.linalg.norm(left - right)
    tolerance = abs(previous-sigma)
    previous = sigma
    print(f"Current iteration {iteration+1} has predicted eigenvalue of {sigma} and residual error of {residual_error} with tolerance {tolerance}"
          f"and the norm of calculated eigenvector is {np.linalg.norm(v)}")

    iteration += 1

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for finding eigenvalue using multigrid is: :", elapsed_time, "seconds")

#error = abs(sigma - correct_answer) / correct_answer * 100
#print(f"The predicted eigenvalue is {sigma} with iteration {iteration} and error {error}%")

left = A @ v
right = sigma * (M @ v)
residual_error = np.linalg.norm(left - right)
print(f"The residual error is {residual_error}")
'''''''''''