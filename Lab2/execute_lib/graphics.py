import numpy as np
from matplotlib import pyplot as plt

from OptimizationMethods.Lab2.execute_lib.help_functions import *


def draw_regression(method, function, start, data, data_real, init_coefs, title=False):
    iter, points = method.execute(start, function)
    xs = [i[0] for i in points[:, 0]]
    ys = [i[1] for i in points[:, 0]]

    vector_of_results = points[-1][0]

    if not title:
        title = method.name+": "+str(iter)
    plt.title(title)

    def result(x):
        res = 0
        for i in range(len(vector_of_results)):
            res += vector_of_results[i] * x ** i
        return res

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    xr = [i[0] for i in data_real]
    yr = [i[1] for i in data_real]
    left, right = min(x), max(x)
    x0 = np.linspace(left - 0.1, right + 0.1, 100)
    y0 = [result(i) for i in x0]

    ax = plt.subplot()
    comp = "$ Computed: " + " + ".join([
            f"{vector_of_results[i]:.3f}" +
            " \cdot x ^ {" + str(i) + "}"
            for i in range(len(vector_of_results))]) + " $"
    init = "$ Initial: " + " + ".join([
            f"{init_coefs[i]:.3f}" +
            " \cdot x ^ {" + str(i) + "}"
            for i in range(len(init_coefs))])+"$"
    ax.plot(x, y, ".")
    ax.plot(xr, yr,"-",label=init)
    ax.plot(x0, y0, "-", label=comp)
    ax.legend(prop='monospace')
    print(init_coefs,";",vector_of_results)

    plt.show()


def draw_levels(function, start, *args, frame=10):

    ax = plt.subplot()
    plt.title(function.get_title())
    min_dots, max_dots = np.math.inf, - np.math.inf
    min_x, max_x = np.math.inf, - np.math.inf
    min_y, max_y = np.math.inf, - np.math.inf

    for i in args:
        iter, points = i.execute(start, function)
        xs = [i[0] for i in points[:, 0]]
        ys = [i[1] for i in points[:, 0]]

        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        min_y = min(min_y, min(ys))
        max_y = max(max_y, max(ys))
        min_dots = min([min_dots, min_x, min_y])
        max_dots = max([max_dots, max_x, max_y])

        ax.plot(xs, ys, '.-', label=i.name + " : " + str(iter) + " : " + f"{points[-1][1]:.2}")
        ax.legend(prop='monospace')

        print(i.name + ";" + str(iter) + ";" + str(iter * 2 + 2) + ";"
              + str(i.get_lrName()) + ";" + str(i.get_lr()) + ";" + str(start) + ";" + str(
            points[-1][1]) + ";" + function.get_title())


    numb = 300

    min_x, max_x = center(min_dots, max_dots, min_x, max_x)
    min_y, max_y = center(min_dots, max_dots, min_y, max_y)
    x = np.linspace(min_x - frame, max_x + frame, numb)
    y = np.linspace(min_y - frame, max_y + frame, numb)
    X, Y = np.meshgrid(x, y)
    Z = function.func([X, Y])
    contour = plt.contour(X, Y, Z)
    plt.clabel(contour, inline=True, fontsize=9)
    plt.colorbar()
    plt.show()


def show_tests_graph(res,
                     title=None,
                     xy_names=None,
                     plot_comment=None,
                     plot_style="-",
                     color="tab:blue",
                     plot_type="plot"):
    xs = np.asarray([i[0] for i in res])
    ys = np.asarray([i[1] for i in res])

    ax = plt.subplot()

    if plot_type == "hist":
        plot = ax.bar(xs, ys, width=0.5, color=color)
        for rect in plot:
            a = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, a, f"{a:.3f}", ha='center', va='bottom')
    else:
        plot, = ax.plot(xs, ys, plot_style, color=color)

    if plot_comment is not None:
        plot.set_label(plot_comment)
        ax.legend(prop='monospace')
    if title is not None:
        plt.title(title)
    if xy_names is not None:
        plt.xlabel(xy_names[0])
        plt.ylabel(xy_names[1])

    plt.show()