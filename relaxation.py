import numpy as np
from sympy import *


def boundary_check(x_inside: float, x_left: float, x_right: float) -> None:
    if not (x_left <= x_inside <= x_right):
        raise ValueError(
            "Starting approximation: {0}, doesn't belong search interval: [{1}, {2}]".format(x_inside, x_left, x_right))


def relaxation(f, fDX, x_left: float, x_right: float, x_start: float, eps: float = 1e-4) -> float:
    boundary_check(x_start, x_left, x_right)

    x_values = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))

    min_value, max_value = np.min(
        np.abs(fDX(x_values))), np.max(np.abs(fDX(x_values)))

    tau = 2. / (min_value + max_value)
    q = (max_value - min_value) / (min_value + max_value)

    x_i, x_prev = x_start, x_start - 2 * eps
    num_iteration = 0

    while abs(x_i - x_prev) >= eps:
        x_i, x_prev = x_i - np.sign(fDX(x_i)) * f(x_i) * tau, x_i
        num_iteration += 1

    estimated_num_iteration = int(
        np.ceil(np.log(np.abs(x_start - x_i) / eps) / np.log(1. / q))) + 1

    print('Relaxtion method. Apriori iteration number: {0}, aposteriori iteration number: {1}'.format(
        estimated_num_iteration, num_iteration))

    return x_i


def main():
    print('Variant 25')

    x = Symbol('x')
    function = np.e ** x - 2*(x-1)**2
    functionDX = function.diff(x)
    f = lambdify(x, function, 'numpy')
    fDX = lambdify(x, functionDX, 'numpy')

    print(relaxation(f,fDX,0,1,0))

    


main()
