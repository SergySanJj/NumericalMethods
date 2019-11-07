import numpy as np
import scipy
from sklearn.metrics import mean_squared_error as mse

eps = 1e-9
max_iter = 1000


def get_dominant_matrix(dim):
    rand = np.random.randn(dim, dim)
    abs = np.abs(rand)
    sums = abs.sum(axis=1) - abs.diagonal()
    add = np.diagflat(sums * np.sign(rand.diagonal()))
    ret = rand + add
    return ret


def solve_g(x, b):
    A = np.hstack((x, b.reshape(b.shape[0], 1)))
    N = x.shape[0]

    swaps = []
    for i in range(0, N):
        # print(A)
        val, ind = select_max(A, N, i)
        if np.abs(val) < eps:
            continue
        swap_columns(A, i, ind)
        swaps.append((i, ind))

        unify_columns(A, i, N)
        for k in range(ind + 1, N):
            A[k] = A[k] - A[i]

    res = back_run(A)

    for i in range(len(swaps) - 1, -1, -1):
        swap = swaps[i]
        swap_columns(res, swap[1], swap[0])
    return res


def back_run(A):
    N = A.shape[0]
    res = np.zeros((1, N))
    print()
    for i in range(N - 1, -1, -1):
        if np.abs(A[i, i]) < eps:
            continue
        res[0, i] = A[i, N] / A[i, i]
        found = res[0, i]
        for k in range(i - 1, -1, -1):
            A[k, N] = A[k, N] - found * A[k, i]
    return res


def swap_rows(X, a, b):
    X[[a, b]] = X[[b, a]]


def swap_columns(X, a, b):
    X[:, [a, b]] = X[:, [b, a]]


def select_max(X, N, row):
    curr_max = 0.
    curr_ind = 0
    for i in range(0, N):
        if np.abs(X[row, i]) > np.abs(curr_max):
            curr_ind = i
            curr_max = X[row, i]
    return (curr_max, curr_ind)


def unify_columns(X, col, N):
    for i in range(col, N):
        if np.abs(X[i, col]) > eps:
            X[i] = X[i] / X[i, col]


def solve_seidel(A, b):
    x = np.random.randn(A.shape[0])
    for _ in range(max_iter):
        x_new = np.zeros(A.shape[0])  # start solution
        for i in range(A.shape[0]):
            one = A[i, :i].dot(x_new[:i])  # sum before i
            two = A[i, i + 1:].dot(x[i + 1:])  # sum after i
            if np.abs(A[i, i]) < eps:
                continue
            x_new[i] = (b[i] - one - two) / A[i, i]
        if mse(x_new, x) < eps:
            break
        x = x_new
    return x


def solve(method: str, A: np.array, b: np.array):
    solution = None
    if method == 'gaussian':
        # solution = solve_gaussian(A, b)
        solution = solve_g(A, b)
    elif method == 'seidel':
        solution = solve_seidel(A, b)
    return solution


def get_matrix(method: str, dim: int):
    m = None
    if method == 'hilbert':
        m = scipy.linalg.hilbert(dim)
    elif method == 'dominant':
        m = get_dominant_matrix(dim)
    elif method == 'symmetric':
        m = np.array([
            [7., 2., 3., 1.],
            [2., 9., 6., 3.],
            [3., 6., 8., 1.],
            [1., 3., 1., 9.]])
    return m


if __name__ == '__main__':
    dim = 4
    matrix_types = ['hilbert', 'dominant', 'symmetric']
    A = get_matrix(matrix_types[2], dim=dim)
    print('A:\n', A)
    det = np.linalg.det(A)
    if np.abs(det)<eps:
        print("Matrix has det 0")
    else:
        print("det(A) = ", det)
        cond = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A))
        print("cond(a) = ", cond)

        # x = np.random.randn(dim)
        x = np.array([1., 2., 3., 4.])
        print('x:\n', x)

        b = A.dot(x)
        print('b:\n', b)
        print()
        for method in ['gaussian', 'seidel']:
            solution = solve(method, A, b)
            print(f'{method} solution:\n', solution)
            actual_b = np.matmul(A, solution.T)
            print('mse:\n', mse(actual_b, b))
            print('\n')
