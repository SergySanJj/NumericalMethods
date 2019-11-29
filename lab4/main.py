import numpy as np
from sympy import *
import matplotlib.pyplot as plt

x_symbol = Symbol('x')
function = (x_symbol**3 - 20*x_symbol**3 * sin(x_symbol))
function_DX_1 = function.diff(x_symbol)
function_DX_2 = function_DX_1.diff(x_symbol)
function_DX_3 = function_DX_2.diff(x_symbol)
function_DX_4 = function_DX_3.diff(x_symbol)
function_DX_5 = function_DX_4.diff(x_symbol)


def delt(yi: float, yi_1: float, xi: float, xi_1: float) -> float:
    return (yi_1 - yi)/(xi_1 - xi)


def Eval(a, x, r):
    ''' a : array returned by function coef()
        x : array of data points
        r : the node to interpolate at '''
    x.astype(float)
    n = len(a) - 1
    temp = a[n] + (r - x[n])
    for i in range(n - 1, -1, -1):
        temp = temp * (r - x[i]) + a[i]
    return temp  # return the y_value interpolation


def coef_newton(x, y):
    '''x : array of data points
       y : array of f(x)  '''
    x.astype(float)
    y.astype(float)
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

    return np.array(a)  # return an array of coefficient


def coef_hermite(x, input):
    n = len(x)

    for i in range(0,n):
        print(np.array(input[round(x[i], 4)]))

    m = -1
    for i in range(0, n):
        m += len(input[round(x[i], 4)])

    print(m)

    a = [[0 for x in range(m+2)] for y in range(m+1)]
    prev = 0
    for i in range(0, n):
        diffs = input[round(x[i], 4)]
        for d in range(0, len(diffs)):
            a[prev][0] = (diffs[0])
            prev += 1

    prev = 1
    for i in range(1, n):
        diffs = input[round(x[i-1], 4)]
        for d in range(0, len(diffs)):
            if len(diffs) > 1:
                a[prev][1] = diffs[1]
            else:
                a[prev][1] = (a[prev][0] - a[prev-1][0])/(x[i] - x[i-1])
                print("HERE")
            prev += 1

    print(np.array(a))


def main():
    eps = 0.01
    x_left = 0.
    x_right = 4.

    f = lambdify(x_symbol, function, 'numpy')
    f_dx1 = lambdify(x_symbol, function_DX_1, 'numpy')
    f_dx2 = lambdify(x_symbol, function_DX_2, 'numpy')
    f_dx3 = lambdify(x_symbol, function_DX_3, 'numpy')
    f_dx4 = lambdify(x_symbol, function_DX_4, 'numpy')
    f_dx5 = lambdify(x_symbol, function_DX_5, 'numpy')

    x = [0.5, 1.0, 3.0]
    input = {
        round(0.5, 4): [f(0.5), f_dx1(0.5)],
        round(1.0, 4): [f(1.0), f_dx1(1.0), f_dx2(1.0), f_dx3(1.0), f_dx4(1.0), f_dx5(1.0)],
        round(3.0, 4): [f(3.0), f_dx1(3.0), f_dx2(3.0), f_dx3(3.0)]
    }

    coef_hermite(x, input)

    xT = np.linspace(x_left, x_right, min(int(1. / eps), 10 ** 5))
    yT = f(xT)
    plt.plot(xT, yT)
    plt.savefig('start.png')


main()
