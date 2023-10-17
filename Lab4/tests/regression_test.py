import numpy as np
import matplotlib.pyplot as plt
import scipy

from functools import partial
from Lab4.util.graphic_util import plot_both_regr, draw_hist
from Lab4.lib_unwrapped.gradient_descent import bfgs, gauss_newton, dog_leg, l_bfgs
from Lab4.regression.regression import gen_points_with_source_f, poly_regression, gen_loss
from Lab4.util.run_functions import run_both_regr, RunResult


def time_memory_test(iters, p, func, func_name):
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
        print(name, end='')
        for _ in range(iters):
            points, src = gen_points_with_source_f(10 + int(np.random.rand() * 30), np.random.rand() * 10, func)
            loss_func = gen_loss(points)
            result_1, result_2 = run_both_regr(p, points, my_regr, scipy_regr, loss_func)
            custom_result.add(result_1)
            scipy_result.add(result_2)
            print('.', end='')  # debug to control test stage
        named_results.append((name, custom_result))
        named_results.append((scipy_name, scipy_result))
        print()
    draw_hist(named_results, lambda x: x[1].time_usage / iters, f'Average time regression for {func_name}')
    draw_hist(named_results, lambda x: x[1].memory_usage / iters, f'Average memory regression for {func_name}')
    draw_hist(named_results, lambda x: x[1].loss / iters, f'Average losses regression for {func_name}')


def main():
    f = lambda x: 5 * x ** 3 - 8 * x ** 2 - x + 9
    p = 3
    points, src = gen_points_with_source_f(30, 3, f)
    all_regrs = [
        (partial(poly_regression, gd=bfgs), partial(scipy.optimize.minimize, method="BFGS"), "BFGS"),
        (partial(poly_regression, gd=l_bfgs), partial(scipy.optimize.minimize, method="L-BFGS-B"), "L-BFGS"),
        (gauss_newton, scipy.optimize.least_squares, "Gauss-Newton"),
        (dog_leg, partial(scipy.optimize.least_squares, method="dogbox"), "Powell's Dog Leg"),
    ]

    for regr in all_regrs:
        my_regr, scipy_regr, name = regr
        plot_both_regr(p, points, plt, my_regr, scipy_regr, name)
        print(name)
        plt.show()

    functions = [
        (lambda x: 5 * x ** 3 - 8 * x ** 2 - x + 9, 3, '5x^3-8x^2-x+9'),
        (lambda x: np.log(5 * x ** 4) - np.exp(x) * np.sqrt(np.exp(x)) - 2 * x ** 2 + 4 * x - 4, 5,
         'log(5x^4)-exp(x)^1.5-2*x^2+4x-5'),
        (lambda x: np.exp(x ** 2) / (8 * x ** 5 - 9), 5, 'exp(x^2)/(8x^5-9)'),
        (lambda x: 4 * (2 * x * np.exp(np.sin(x ** 4)) + 3 * x ** 2), 4, '8x*exp(sin(x^4))+3*x^2')
    ]

    for func, p, name in functions:
        time_memory_test(50, p, func, name)


if __name__ == "__main__":
    main()
