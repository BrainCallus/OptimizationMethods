import numpy as np

from OptimizationMethods.Lab2.lib.errors_functions import quadratic_error_func, quadratic_error_func_grad
from OptimizationMethods.Lab2.lib.functions_and_gradients import MiniBatchGD
from OptimizationMethods.Lab2.lib.polynom_function import polynom
from OptimizationMethods.Lab2.execute_lib.regression_generation import generate_descent_polynom

def do_several_tests(test_func, n, *args):
    res = []
    for i in range(n):
        res.append(test_func(*args))
    return np.mean(np.asarray(res), axis=0)


def batch_size_test(method, start, finish, data_size):
    start_point = [0, 0]
    xs, ys, y_real = generate_descent_polynom(15, polynom([np.random.uniform(0, 10), np.random.uniform(0, 10)]),
                                              data_size)

    xy = np.dstack((xs, ys))[0]
    res = []

    func = MiniBatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

    for i in range(start, finish):
        j = i + 1
        func.set_batch(j)
        iterations, _ = method.execute(start_point, func)
        res.append([j, iterations])

    return res