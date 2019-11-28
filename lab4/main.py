import numpy as np
from sympy import *
import matplotlib.pyplot as plt

x_symbol = Symbol('x')
function = (x_symbol**3 - x_symbol**3 * sin(x_symbol))
function_DX_1 = function.diff(x_symbol)
function_DX_2 = function_DX_1.diff(x_symbol)
function_DX_3 = function_DX_2.diff(x_symbol)
function_DX_4 = function_DX_3.diff(x_symbol)
function_DX_5 = function_DX_4.diff(x_symbol)

print(function_DX_1)


def delt(yi: float, yi_1: float, xi: float, xi_1: float) -> float:
    return (yi_1 - yi)/(xi_1 - xi)


def delt(X, Y):
    if


def main():
    eps = 0.01
    x_left = 0.
    x_right = 20.

    f = lambdify(x_symbol, function, 'numpy')
    f_dx1 = lambdify(x_symbol, function_DX_1, 'numpy')
    f_dx2 = lambdify(x_symbol, function_DX_2, 'numpy')
    f_dx3 = lambdify(x_symbol, function_DX_3, 'numpy')
    f_dx4 = lambdify(x_symbol, function_DX_4, 'numpy')
    f_dx5 = lambdify(x_symbol, function_DX_5, 'numpy')

    x = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))
    y = f(x)

    y_dx1 = f_dx1(x)
    y_dx2 = f_dx2(x)
    y_dx3 = f_dx3(x)
    y_dx4 = f_dx4(x)
    y_dx5 = f_dx5(x)

    plt.plot(x, y)
    # plt.plot(x, y_dx1)
    # plt.plot(x, y_dx2)
    # plt.plot(x, y_dx3)
    # plt.plot(x, y_dx4)
    # plt.plot(x, y_dx5)
    plt.savefig('start.png')


main()


# def hermit_interpolate(input):  #input is list of tuples [(x1,y1),(x2,y2),...,(xn,yn)] xi are Chebyshev nodes
#     n = len(input)
#     points = np.zeros(shape=(2 * n + 1, 2 * n + 1))
#     X, Y = zip(*input)
#     X = list(X)
#     Y = list(Y)

#     for i in range(0, 2 * n, 2):
#         points[i][0] = X[i / 2]
#         points[i + 1][0] = X[i / 2]
#         points[i][1] = Y[i / 2]
#         points[i + 1][1] = Y[i / 2]

#     for i in range(2, 2 * n + 1):
#         for j in range(1 + (i - 2), 2 * n):
#             if i == 2 and j % 2 == 1:
#                 points[j][i] = calculate_f_p_x(X[j / 2]);

#             else:
#                 points[j][i] = (points[j][i - 1] - points[j - 1][i - 1]) / (
#                     points[j][0] - points[(j - 1) - (i - 2)][0])

#     def result_polynomial(xpoint):  #here is function to calculate value for given x
#         val = 0
#         for i in range(0, 2 * n):
#             factor = 1.
#             j = 0
#             while j < i:
#                 factor *= (xpoint - X[j / 2])
#                 if j + 1 != i:
#                     factor *= (xpoint - X[j / 2])
#                     j += 1
#                 j += 1
#             val += factor * points[i][i + 1]
#         return val
