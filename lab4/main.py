import numpy as np
from sympy import *
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

x_symbol = Symbol('x')
# function =  (x_symbol ** 3 - 20 * x_symbol ** 3 * sin(x_symbol) + np.e**x_symbol)
function = sin(x_symbol)*x_symbol
function_DX_1 = function.diff(x_symbol)
function_DX_2 = function_DX_1.diff(x_symbol)
function_DX_3 = function_DX_2.diff(x_symbol)
function_DX_4 = function_DX_3.diff(x_symbol)
function_DX_5 = function_DX_4.diff(x_symbol)


def delt(yi: float, yi_1: float, xi: float, xi_1: float) -> float:
    return (yi_1 - yi) / (xi_1 - xi)


def eval(coeffs, a, r):
    n = len(coeffs)
    carry = 1.
    res = 0.
    for i in range(0, n):
        res += coeffs[i] * carry
        carry *= r - a[i][0]

    left = coeffs[n - 1] * carry
    # print("R: ", left)

    return res, left


def coef_hermite(x, inp):
    n = len(x)

    m = -1
    for i in range(0, n):
        m += len(inp[round(x[i], 4)])

    print("m: ", m)

    a = [[0 for x in range(m + 2)] for y in range(m + 1)]
    prev = 0
    for i in range(0, n):
        diffs = inp[round(x[i], 4)]
        for d in range(0, len(diffs)):
            a[prev][0] = (x[i])
            a[prev][1] = (diffs[0])
            prev += 1

    for deep in range(1, m):

        for i in range(deep, m):
            xv = a[i][0]
            diffs = inp[round(xv, 4)]
            x_start = a[i - deep][0]
            if round(xv, 4) == round(x_start, 4) and len(diffs) > deep:
                a[i][deep + 1] = diffs[deep] / math.factorial(deep)
            else:
                a[i][deep + 1] = delt(a[i][deep], a[i - deep][deep], xv, x_start)

    # print(np.array(a))
    coeffs = []
    for i in range(0, m):
        coeffs.append(a[i][i + 1])

    return (coeffs, a)


def main():
    eps = 0.001
    x_left = 0.
    x_right = 15. - eps

    f = lambdify(x_symbol, function, 'numpy')
    f_dx1 = lambdify(x_symbol, function_DX_1, 'numpy')
    f_dx2 = lambdify(x_symbol, function_DX_2, 'numpy')
    f_dx3 = lambdify(x_symbol, function_DX_3, 'numpy')
    f_dx4 = lambdify(x_symbol, function_DX_4, 'numpy')
    f_dx5 = lambdify(x_symbol, function_DX_5, 'numpy')

    inp = {
        round(0.5, 4): [f(0.5), f_dx1(0.5)],
        round(10.0, 4): [f(10.0), f_dx1(10.0), f_dx2(10.0), f_dx3(10.0), f_dx4(10.0), f_dx5(10.0)],

        round(15., 4): [f(15.)],
        round(20.0, 4): [f(20.), f_dx1(20.), f_dx2(20.), f_dx3(20.), f_dx4(20.), f_dx5(20.)],
    }

    x = []
    for key in inp:
        x.append(round(key, 4))

    xT = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))
    yT = f(xT)
    plt.plot(xT, yT)

    herm, a = coef_hermite(x, inp)
    print("coeffs: ", np.array(herm))

    yT_inter = []
    Rs = 0.
    for dp in xT:
        val, r = eval(herm, a, dp)
        yT_inter.append(val)
        Rs += np.abs(r)

    yT_inter = np.array(yT_inter)
    plt.plot(xT, yT_inter)

    plt.savefig('start.png')

    print("R sum: ", Rs)
    # print("MSE: ", mean_squared_error(yT, yT_inter))
    print("Finished")


main()
