import numpy as np
import matplotlib.pyplot as plt
import scipy

from functools import partial
from Lab4.lib_unwrapped.gradient_descent import gauss_newton, dog_leg
from Lab4.regression.regression import gen_loss, gen_points_with_source_f
from Lab4.util.run_functions import run_both_regr

# Тут хочется на одном графике несколько линий зависимости параметров(память/время/потери) от границ

def time_menory_test(bounds, func, lf, iters, p):
    points, src = gen_points_with_source_f(10 + int(np.random.rand() * 30), np.random.rand() * 10, func)
    loss_fun = lf(points)
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    for bound in bounds:
        print(f"Bound = [-{bound}, {bound}]:")
        all_regrs = [
            (gauss_newton, partial(scipy.optimize.least_squares, bounds=(-bound, bound)), "Gauss-Newton"),
            (dog_leg, partial(scipy.optimize.least_squares, method="dogbox", bounds=(-bound, bound)),
             "Powell's Dog Leg"),
        ]
        for regr, ax in zip(all_regrs, axs.flatten()):
            my_regr, scipy_regr, name = regr
            losses1 = losses2 = time_acc1 = time_acc2 = mem_acc1 = mem_acc2 = 0
            for _ in range(iters):
                result_1, result_2 = run_both_regr(p, points, lambda _, __: (), scipy_regr)
                losses1 += loss_fun(result_1.result)
                losses2 += loss_fun(result_2.result)
                time_acc1 += result_1.time_usage
                time_acc2 += result_2.time_usage
                mem_acc1 += result_1.memory_usage
                mem_acc2 += result_2.memory_usage

            print(f"""\t{name}:\n"""
                  f"""\t\tLoss: {losses1 / iters} vs {losses2/iters}\n"""
                  f"""\t\tMemory: {mem_acc1 / iters} vs {mem_acc2/iters}\n"""
                  f"""\t\tTime: {time_acc1 / iters} vs {time_acc2/iters}""")


def main():
    p = 3
    iters = 100
    f = lambda x: 3 + 3 * x + 3 * x ** 2

    bounds = [1, 3, 5, 10, 100]
    time_menory_test(bounds, f, gen_loss, iters, p)


if __name__ == "__main__":
    main()
