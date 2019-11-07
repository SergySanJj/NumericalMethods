import numpy as np

# chose by rows


def solve(x, b, N):
    A = np.hstack((x, b.T))
    for i in range(0, N):
        print(A)
        val, ind = select_max(A, N, i)

        unify_columns(A, ind, N)

        curr_row = A[i]

        for k in range(0, N):
            if k != i:
                A[k] = A[k]-A[i]

        #swap_rows(A, i, ind)

    return A


def swap_rows(X, a, b):
    X[[a, b]] = X[[b, a]]


def select_max(X, N, row):
    curr_max = 0.
    curr_ind = 0
    for i in range(0, N):
        if np.abs(X[row, i]) > np.abs(curr_max):
            curr_ind = i
            curr_max = X[row, i]
    return (curr_max, curr_ind)


def unify_columns(X, col, N):
    for i in range(0, N):
        X[i] = X[i] / X[i, col]


def main():
    N = 3
    x = np.matrix([[1., 2., 4.],
                   [4., 5., 6.],
                   [7., 8., 9.]])

    b = np.matrix([2., 3., 5.])

    print(solve(x, b, N))


main()
