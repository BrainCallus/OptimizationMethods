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


def draw_regression(data, vector_of_results, title=False):
    if not title:
        title = "$" + " + ".join([
                    f"{vector_of_results[i]:.3f}" +
                    " \cdot x ^ {" + str(i) + "}"
                    for i in range(len(vector_of_results))]) + " $"
    plt.title(title)

    def result(x):
        res = 0
        for i in range(len(vector_of_results)):
            res += vector_of_results[i] * x ** i
        return res

    xs = [i[0] for i in data]
    ys = [i[1] for i in data]
    left, right  = min(xs), max(xs)
    x0 = np.linspace(left - 1, right + 1, 100)
    y0 = [result(i) for i in x0]

    plt.plot(xs, ys, ".")
    plt.plot(x0, y0, "-")

    plt.show()