import math

import numpy as np
import random
import scipy as sp

import set_up


# Initialize eigenvector Guesses, Return X or False
def generate_initial_vectors(n, p, M):
    # Initial Guess p number of eigenvectors
    X = np.zeros((n, p))

    for i in range(p):
        np.random.seed(i)
        X[:, i] = np.random.rand(n)

        count = 1  # generate new seed

        iteration = 0
        MAXINTERATION = 10
        while np.linalg.matrix_rank(X) < i and iteration < MAXINTERATION:
            np.random.seed(i + p * count)
            X[:, i] = np.random.rand(n)
            iteration += 1
    if np.linalg.matrix_rank(X) == p:
        X = gram_schmidt(X, M,0)
        return X
    else:
        print("Error: Unable to generate linearly independent vectors")
        return False


# Gram_Schmidt Process, k: starting location
def gram_schmidt(X, M, start):
    n = X.shape[1]  # number of vectors

    for i in range(start,n):
        v_current = X[:, i]
        u_current = v_current
        for j in range(i):
            u_current = u_current - (v_current.T @ M @ X[:, j]) * X[:, j]

        # Error for zero vectors
        if np.all(np.isclose(u_current, 0, atol=10 ** -5)):
            print("Error")
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        norm = u_current.T @ M @ u_current
        norm = math.sqrt(norm)
        u_current = u_current / norm
        X[:, i] = u_current

    return X


# Generate Basis
def generate_basis_V(n, p, X,M):
    W = np.random.randn(n, p)
    V_main = np.hstack((X, W))

    start = X.shape[1]

    #V, _ = np.linalg.qr(V_main)
    V = gram_schmidt(V_main, M, start)
    W = V[:,start:]
    #k = np.linalg.matrix_rank(V)
    return W,X

def form_matrix (X,W,A):
    xax = X.T @ A @ X
    xaw = X.T @ A @ W
    wax = xaw.T
    waw = W.T @ A @ W

    matrix1 = np.hstack((xax,xaw))
    matrix2 = np.hstack((wax,waw))

    matrix = np.vstack((matrix1,matrix2))
    return matrix


# Completely Solve 2p number of eigenvectors in V.TAV x = l V.TMV x, return actual 2p of eigenvectors of A
def solve_eigen(A, M, p, X):
    m, n = A.shape
    W,X = generate_basis_V(n, p, X,M)

    A_new = form_matrix (X,W,A)
    M_new = form_matrix (X,W,M)

    _, eigenvectors = sp.linalg.eig(A_new, M_new)  # return normalized eigenvectors

    alpha = eigenvectors[:p,:]
    beta = eigenvectors[p:,:]
    X = X @ alpha + W @ beta

    return X


# Apply Rayleigh Quotient to find eigenvalues
def find_eigenvalue(X, A, M):
    n = X.shape[1]
    eigenvalues = np.zeros(n)
    for i in range(n):
        x = X[:, i]
        l = (x.T @ A @ x) / (x.T @ M @ x)
        eigenvalues[i] = l

    sorted_indices = np.argsort(eigenvalues)  # get index
    eigenvalues = np.sort(eigenvalues)
    return sorted_indices, eigenvalues


# Get new X
def pick_column(X, indices, p):
    get_indices = indices[:p]
    X = X[:, get_indices]
    return X


total_size = 100
p = 5
adj, A, M = set_up.make_graph_2(total_size)


# Actual Answer
eigenvalues, eigenvectors = sp.linalg.eig(A, M)
v = np.sort(eigenvalues)
print(v[:p])


# Working basic calculations
def basic_calculation(total_size, p, A, M):
    X = generate_initial_vectors(total_size, p, M)
    iteration = 0
    error = 10
    v_p_old = 10
    while iteration < 10000 and error > 10 ** -10:
        X = solve_eigen(A, M, p, X)
        sorted_indices, eigenvalues = find_eigenvalue(X, A, M)
        v_calc = np.sort(eigenvalues)
        X = pick_column(X, sorted_indices, p)

        v_new = v_calc[p - 1]
        error = abs(v_p_old - v_new)
        v_p_old = v_new
        iteration += 1
    print(eigenvalues[:p])
    print(f"Number of Iteration is {iteration}")


basic_calculation(total_size, p, A, M)



