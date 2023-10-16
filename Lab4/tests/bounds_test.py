import numpy as np
import matplotlib.pyplot as plt
import scipy

from functools import partial
from Lab4.lib_unwrapped.gradient_descent import gauss_newton, dog_leg
from Lab4.regression.regression import gen_loss, gen_points_with_source_f
from Lab4.util.graphic_util import draw_dependence_graph
from Lab4.util.run_functions import run_both_regr, RunResult


# Тут хочется на одном графике несколько линий зависимости параметров(память/время/потери) от границ


def time_menory_test(bounds, func, iters, p):
    points, src = gen_points_with_source_f(10 + int(np.random.rand() * 30), np.random.rand() * 10, func)
    loss_fun = gen_loss(points)
    results_by_name = {
        'Gauss-Newton': [], 'Least Squares': [], 'Powell\'s Dog Leg': [], 'Least squares dogbox': []
    }
    for bound in bounds:
        print(f"Bound = [-{bound}, {bound}]:")
        all_regrs = [
            (gauss_newton, partial(scipy.optimize.least_squares, bounds=(-bound, bound)), "Gauss-Newton",
             "Least Squares"),
            (dog_leg, partial(scipy.optimize.least_squares, method="dogbox", bounds=(-bound, bound)),
             "Powell's Dog Leg", "Least squares dogbox"),
        ]
        for regr in all_regrs:
            custom_result = RunResult(None, 0, 0)
            torch_result = RunResult(None, 0, 0)
            my_regr, scipy_regr, custom_name, torch_name = regr
            for _ in range(iters):
                result_1, result_2 = run_both_regr(p, points, my_regr, scipy_regr, loss_fun)
                custom_result.add(result_1)
                torch_result.add(result_2)

            results_by_name.get(custom_name).append(custom_result)
            results_by_name.get(torch_name).append(torch_result)

    draw_dependence_graph('Time', lambda res: res.time_usage / iters, results_by_name, bounds)
    draw_dependence_graph('Memory kb', lambda res: res.memory_usage / iters, results_by_name, bounds)
    draw_dependence_graph('Losses', lambda res: res.loss / iters, results_by_name, bounds)


def main():
    p = 3
    iters = 20
    f = lambda x: 3 + 3 * x + 3 * x ** 2

    bounds = range(5, 100, 5)
    # for i in range(5, 100, 5):
    time_menory_test(bounds, f, iters, p)


if __name__ == "__main__":
    main()
