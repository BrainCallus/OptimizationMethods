import numpy as np
import matplotlib.pyplot as plt
import scipy

from functools import partial
from Lab4.util.graphic_util import plot_both_regr
from Lab4.lib_unwrapped.gradient_descent import bfgs, gauss_newton, dog_leg, l_bfgs
from Lab4.regression.regression import gen_points_with_source_f, poly_regression
from Lab4.util.run_functions import run_both_regr

# Тут хочется гитограммку отдельно для времени, отдельно для памяти
def time_memory_test(iters, p):
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    all_regrs = [
        (partial(poly_regression, gd=bfgs), partial(scipy.optimize.minimize, method="BFGS"), "BFGS"),
        (partial(poly_regression, gd=l_bfgs), partial(scipy.optimize.minimize, method="L-BFGS-B"), "L-BFGS"),
        (gauss_newton, scipy.optimize.least_squares, "Least Squares"),
        (dog_leg, partial(scipy.optimize.least_squares, method="dogbox"), "Powell's Dog Leg"),
    ]

    for regr, ax in zip(all_regrs, axs.flatten()):
        my_regr, scipy_regr, name = regr
        time_acc1 = time_acc2 = mem_acc1 = mem_acc2 = 0
        for _ in range(iters):
            points, src = gen_points_with_source_f(10 + int(np.random.rand() * 30), np.random.rand() * 10, np.exp)
            result_1, result_2 = run_both_regr(p, points, my_regr, scipy_regr)
            time_acc1 += result_1.time_usage
            time_acc2 += result_2.time_usage
            mem_acc1 += result_1.memory_usage
            mem_acc2 += result_2.memory_usage

        print(f"""{name}:\n"""
              f"""\tMemory: {mem_acc1 / iters} vs {mem_acc2 / iters}\n"""
              f"""\tTime: {time_acc1 / iters} vs {time_acc2 / iters}""")


def main():
    p = 1
    points, src = gen_points_with_source_f(40, 1, np.exp)
    plot_both_regr(p, points, plt, partial(poly_regression, gd=bfgs), scipy.optimize.minimize, "")
    plt.show()
    p = 3
    points, src = gen_points_with_source_f(30, 1, np.exp)
    all_regrs = [
        (partial(poly_regression, gd=bfgs), partial(scipy.optimize.minimize, method="BFGS"), "BFGS"),
        (partial(poly_regression, gd=l_bfgs), partial(scipy.optimize.minimize, method="L-BFGS-B"), "L-BFGS"),
        (gauss_newton, scipy.optimize.least_squares, "Gauss-Newton"),
        (dog_leg, partial(scipy.optimize.least_squares, method="dogbox"), "Powell's Dog Leg"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 12))
    fig.tight_layout()

    for regr, ax in zip(all_regrs, axs.flatten()):
        my_regr, scipy_regr, name = regr
        plot_both_regr(p, points, ax, my_regr, scipy_regr, name)
    plt.show()
    time_memory_test(50, 2)


if __name__ == "__main__":
    main()
