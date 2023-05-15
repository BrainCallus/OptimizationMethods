import numpy as np

from OptimizationMethods.Lab2.documentation.errors_functions import quadratic_error_func, quadratic_error_func_grad
from OptimizationMethods.Lab2.documentation.functions_and_gradients import MiniBatchGD
from OptimizationMethods.Lab2.documentation.polynom_function import polynom
from OptimizationMethods.Lab2.execute_documetation.regression_generation import generate_descent_polynom


def batch_size_test(method, start, finish,  data_size):
    start_point = [0, 0]
    xs, ys, y_real = generate_descent_polynom(15, polynom([np.random.uniform(0, 10), np.random.uniform(0, 10)]),
                                              data_size)
    res = []

    xy = np.dstack((xs, ys))[0]

    func = MiniBatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

    for i in range(start, finish):
        j = i + 1
        func.set_batch(j)
        iterations, _ = method.execute(start_point, func)
        res.append([j, iterations])

    return res