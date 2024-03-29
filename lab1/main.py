import numpy as np
from sympy import *


def boundary_check(x_inside, x_left, x_right) -> None:
    if not (x_left <= x_inside <= x_right):
        raise ValueError(
            "{0} not in [{1}, {2}]".format(x_inside, x_left, x_right))

def relaxation(f, fDX, x_left, x_right, x_start, eps=1e-4):
    boundary_check(x_start, x_left, x_right)

    x_values = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))

    m1, M1 = np.min(
        np.abs(fDX(x_values))), np.max(np.abs(fDX(x_values)))

    tau = 2. / (m1 + M1)
    q = (M1 - m1) / (m1 + M1)

    x_i, x_prev = x_start, x_start - 2 * eps
    num_iteration = 0

    while abs(x_i - x_prev) >= eps:
        x_i, x_prev = x_i - np.sign(fDX(x_i)) * f(x_i) * tau, x_i
        num_iteration += 1

    estimated_num_iteration = int(
        np.ceil(np.log(np.abs(x_start - x_i) / eps) / np.log(1. / q))) + 1

    print('Relaxation method. \nApriori steps: {0}, aposteriori steps: {1}'.format(
        estimated_num_iteration, num_iteration))

    return x_i


def newton(f, fDX, x_left, x_right, x_start, eps=1e-4):
    boundary_check(x_start, x_left, x_right)

    num_iteration = 0
    x_values = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))

    m1, M1 = np.min(
        np.abs(fDX(x_values))), np.max(np.abs(fDX(x_values)))

    q = (M1 - m1) / (m1 + M1)

    x_i, x_prev = x_start, x_start - 2 * eps
    num_iteration = 0

    while abs(x_i - x_prev) >= eps:
        x_i, x_prev = x_i - f(x_i)/fDX(x_i), x_i
        num_iteration += 1

    estimated_num_iteration = int(np.ceil(
        np.log2(np.abs(np.log(np.abs(x_start - x_i) / 2) / np.log(1. / q) + 1)))) + 1

    print('Newton\'s method. \nApriori steps: {0}, aposteriori steps: {1}'.format(
        estimated_num_iteration, num_iteration))
    return x_i


def main():
    print('Variant 25')

    x = Symbol('x')
    function = np.e ** x - 2*(x-1)**2
    functionDX = function.diff(x)
    f = lambdify(x, function, 'numpy')
    fDX = lambdify(x, functionDX, 'numpy')

    print(relaxation(f, fDX, 0, 1, 0))
    print(newton(f, fDX, 0, 1, 0))


main()
