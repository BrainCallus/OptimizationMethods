import numpy as np
import matplotlib.pyplot as plt
import math

from Lab4.util.run_functions import run_cusom_and_torch, run_both_regr


def get_plot_bounds(*point_arrs):
    inf = float('inf')
    left_min, right_max, top_max, bottom_max = inf, -inf, -inf, inf
    for points in point_arrs:
        for p in points:
            left_min = min(left_min, p[0])
            right_max = max(right_max, p[0])
            top_max = max(top_max, p[1])
            bottom_max = min(bottom_max, p[1])

    return left_min - 0.5, right_max + 0.5, top_max + 0.5, bottom_max - 0.5


def make_linspace(*point_arrs):
    left, right, top, bottom = get_plot_bounds(*point_arrs)
    X, Y = np.meshgrid(np.linspace(left, right, 100), np.linspace(bottom, top, 100))
    return X, Y


def plot_both(x, f, ax, custom_func, torch_func, name):
    result_1, result_2 = run_cusom_and_torch(x, f, custom_func, torch_func)
    X, Y = make_linspace(result_1.result, result_2.result)
    left, right, top, bottom = get_plot_bounds(result_1.result, result_2.result)
    ax.contour(X, Y, f([X, Y]), levels=[f(np.array([p, p])) for p in
                                        range(1,
                                              math.ceil(min(max(abs(left), abs(right), abs(top), abs(bottom)), 20)))])
    ax.plot(result_1.result[:, 0], result_1.result[:, 1], 'o-')
    ax.plot(result_2.result[:, 0], result_2.result[:, 1], 'o-')
    title = f"{name}\nour {len(result_1.result)} iterations / torch {len(result_2.result)} iterations"
    if ax == plt:
        ax.title(title)
    else:
        ax.set_title(title)


def restore_coefs(c, x):
    return sum(x ** i * c[i] for i in range(len(c)))


def plot_both_regr(p, points, ax, my_impl, scipy_impl, name):
    ax.scatter(points[:, 0], points[:, 1])
    c1, c2 = run_both_regr(p, points, my_impl, scipy_impl)
    X = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)

    def gen_restore(c):
        return lambda x: restore_coefs(c, x)

    ax.plot(X, gen_restore(c1.result)(X))
    ax.plot(X, gen_restore(c2.result)(X))

    title = f"{name}"
    if ax == plt:
        ax.title(title)
    else:
        ax.set_title(title)


def draw_hist(named_results, mapper, title):
    ax = plt.subplot(211)
    width = 0.8
    data = list(map(mapper, named_results))
    bins = list(map(lambda x: x + 1, range(0, len(data))))
    ax.bar(bins, data, width=width)
    ax.set_xticks(list(map(lambda x: x, range(1, len(data) + 1))))
    ax.set_title(title)
    ax.set_xticklabels(list(map(lambda x: x[0], named_results)), rotation=45, rotation_mode="anchor", ha="right")
    plt.show()


def draw_dependence_graph(parameter_name, mapper, results_by_name, bounds):
    plt.figure()
    for item in results_by_name.items():
        times = list(map(mapper, item[1]))
        plt.plot(bounds, times, label=item[0])

    plt.title(f'{parameter_name} dependence on bounds')
    plt.xlabel("bounds")
    plt.ylabel(f'{parameter_name}')
    plt.legend()
    plt.show()
