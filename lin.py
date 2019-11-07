import numpy as np
import scipy
from sklearn.metrics import mean_squared_error as mse

eps = 1e-9

def get_dominant_matrix(dim):
    rand = np.random.randn(dim, dim)
    abs = np.abs(rand)
    sums = abs.sum(axis=1) - abs.diagonal()
    add = np.diagflat(sums * np.sign(rand.diagonal()))
    ret = rand + add
    return ret


def get_matrix(method: str, dim: int):
    m = None
    if method == 'hilbert':
        m = scipy.linalg.hilbert(dim)
    elif method == 'dominant':
        m = get_dominant_matrix(dim)
    elif method == 'random':
        m = np.random.randn(dim, dim)

    return m


def direct_substitution(A, b):
    x = np.empty_like(b)
    x[0] = b[0] / A[0, 0]
    for i in range(1, b.shape[0]):
        x[i] = (b[i] - np.inner(x[:i], A[i, :i])) / A[i, i]
    return x


def inverse_lower_triangular(A):
    B = np.eye(A.shape[0])
    return np.hstack([direct_substitution(A, B[:, i]).reshape(-1, 1) for i in range(A.shape[1])])


def inverse_substitution(A, b):
    x = np.empty_like(b)
    x[-1] = b[-1] / A[-1, -1]
    for i in range(b.shape[0] - 2, -1, -1):
        x[i] = (b[i] - np.inner(x[i + 1:], A[i, i + 1:])) / A[i, i]
    return x



def solve_gaussian(A, b):
    L, U = lu_decomposition(A)
    y = direct_substitution(L, b)
    x = inverse_substitution(U, y)
    return x


max_iter = 1000


def solve_seidel(A, b):
    M_inv = inverse_lower_triangular(np.tril(A))
    N = - np.tril(A.T, -1).T
    return simple_iteration(A, b, M_inv, N)



def solve(method: str, A: np.array, b: np.array):
    solution = None
    if method == 'gaussian':
        solution = solve_gaussian(A, b)
    elif method == 'seidel':
        solution = solve_seidel(A, b)
    return solution


def main():
    dim = 3

    A = get_matrix('dominant', dim=dim)  # hilbert, dominant, random
    print('A:\n', A)

    x = np.random.randn(dim)
    print('x:\n', x)

    b = A.dot(x)
    print('b:\n', b)

    for method in ['gaussian', 'seidel']:
        try:
            solution = solve(method, A, b)
            print(f'{method} solution:\n', solution)
            print('mse:\n', mse(A.dot(solution), b))
            print('\n')
        except Exception as ex:
            print(f'{method} raised: {ex}')

main()
