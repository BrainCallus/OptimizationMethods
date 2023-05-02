import numpy as np
from matplotlib import pyplot as plt


def execute(method, func, start, dim):
    if start is None:
        start = np.full(dim, 100)

    return method.execute(start, func)


def output_LRate(method, func, start=None):

    iterations, points = execute(method, func, start, 2)


    xs = [i[0][0] for i in points]
    ys = [i[0][1] for i in points]

    plt.plot(xs, ys, 'o-')
    print(iterations)
    print(points[-1])
    plt.show()

def drawGraph(method, func, start=None):

    iterations, points = execute(method, func, start, 1)

    xs = [i[0][0] for i in points]
    ys = [i[1] for i in points]

    left, right  = min(xs), max(xs)
    x0 = np.linspace(left - 1, right + 1, 100)
    y0 = [func.func([i]) for i in x0]


    plt.plot(x0, y0, '-')
    plt.plot(xs, ys, '.-', color='red')
    print(iterations)
    print(points[-1])
    plt.show()