import numpy as np
import matplotlib.pyplot as plt
import scipy

from functools import partial
from Lab4.util.graphic_util import plot_both_regr, draw_hist
from Lab4.lib_unwrapped.gradient_descent import bfgs, gauss_newton, dog_leg, l_bfgs
from Lab4.regression.regression import gen_points_with_source_f, poly_regression, gen_loss
from Lab4.util.run_functions import run_both_regr, RunResult


# Тут хочется гитограммку отдельно для времени, отдельно для памяти
def time_memory_test(iters, p, func):
    all_regrs = [
        (partial(poly_regression, gd=bfgs), partial(scipy.optimize.minimize, method="BFGS"), "BFGS", "BFGS(scipy)"),
        (partial(poly_regression, gd=l_bfgs), partial(scipy.optimize.minimize, method="L-BFGS-B"), "L-BFGS",
         "L-BFGS-B(scipy)"),
        (gauss_newton, scipy.optimize.least_squares, "Gauss-Newton", "Least Squares"),
        (dog_leg, partial(scipy.optimize.least_squares, method="dogbox"), "Powell's Dog Leg", "Least squares dogbox")
    ]
    named_results = []

    for regr in all_regrs:
        my_regr, scipy_regr, name, scipy_name = regr
        custom_result = RunResult(None, 0, 0)
        scipy_result = RunResult(None, 0, 0)
        for _ in range(iters):
            points, src = gen_points_with_source_f(10 + int(np.random.rand() * 30), np.random.rand() * 10, func)
            loss_func = gen_loss(points)
            result_1, result_2 = run_both_regr(p, points, my_regr, scipy_regr, loss_func if (
                    name == "Gauss-Newton" or name == "Powell's Dog Leg") else None)
            custom_result.add(result_1)
            scipy_result.add(result_2)
        named_results.append((name, custom_result))
        named_results.append((scipy_name, scipy_result))
        print(name)
    draw_hist(named_results, lambda x: x[1].time_usage / iters, f'Average time regression')
    draw_hist(named_results, lambda x: x[1].memory_usage / iters, f'Average memory regression')
    draw_hist(named_results, lambda x: x[1].loss / iters, f'Average losses regression')


def main():
    # p = 1
    # points, src = gen_points_with_source_f(40, 1, np.exp)
    # plot_both_regr(p, points, plt, partial(poly_regression, gd=bfgs), scipy.optimize.minimize, "")
    # plt.show()
    # p = 3
    # points, src = gen_points_with_source_f(30, 1, np.exp)
    # all_regrs = [
    #    (partial(poly_regression, gd=bfgs), partial(scipy.optimize.minimize, method="BFGS"), "BFGS"),
    #    (partial(poly_regression, gd=l_bfgs), partial(scipy.optimize.minimize, method="L-BFGS-B"), "L-BFGS"),
    #    (gauss_newton, scipy.optimize.least_squares, "Gauss-Newton"),
    #    (dog_leg, partial(scipy.optimize.least_squares, method="dogbox"), "Powell's Dog Leg"),
    # ]
    #
    # fig, axs = plt.subplots(2, 2, figsize=(10, 12))
    # fig.tight_layout()
    #
    # for regr, ax in zip(all_regrs, axs.flatten()):
    #    my_regr, scipy_regr, name = regr
    #    plot_both_regr(p, points, ax, my_regr, scipy_regr, name)
    # plt.show()
    f = lambda x: np.exp(x ** 2) / (8 * x ** 5 - 9)
    time_memory_test(5, 2, f)


if __name__ == "__main__":
    main()
