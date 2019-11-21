import numpy as np


def find_max_eig(A, eps):
    x_k = np.random.rand(A.shape[1])

    curr = 2*eps
    prev = 0.
    while np.abs(curr-prev) > eps:
        e_k = x_k / np.linalg.norm(x_k)
        x_k = A @ e_k
        mu = np.dot(x_k, e_k)

        prev = curr
        curr = mu
    return curr




def find_min_eig(A, eps):
    norm_A = np.linalg.norm(A)
    B = norm_A*np.eye(A.shape[1]) - A
    lmin = norm_A - find_max_eig(B, eps)
    return lmin


def min_max_eig(A, eps):
    lmax = find_max_eig(A, eps)

    B = lmax*np.eye(A.shape[1]) - A
    lmin = lmax - find_max_eig(B, eps)

    return lmin, lmax


def main():
    eps = 1e-12
    A = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ])

    print(A)

    lreal = np.linalg.eigvals(A)
    print('expected min {0}; max {1}'.format(np.min(lreal), np.max(lreal)))

    lmin, lmax = min_max_eig(A, eps)
    print('actual min   {0}; max {1}'.format(lmin, lmax))


main()
