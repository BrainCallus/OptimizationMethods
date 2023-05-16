import time

import numpy as np

from OptimizationMethods.Lab2.lib.errors_functions import quadratic_error_func, quadratic_error_func_grad
from OptimizationMethods.Lab2.lib.functions_and_gradients import MiniBatchGD
from OptimizationMethods.Lab2.lib.polynom_function import polynom
from OptimizationMethods.Lab2.execute_lib.regression_generation import generate_descent_polynom
from OptimizationMethods.Lab2.lib.regularization import *


def do_several_tests(test_func, n, *args):
    res = []
    for i in range(n):
        res.append(test_func(*args))
    return np.mean(np.asarray(res), axis=0)

def do_several_tests_with_consts(test_func, n, *args):
    if n == 0: return None

    res = []
    names = []
    for i in range(n):
        press = test_func(*args)
        names = [i[0] for i in press]
        value = [i[1] for i in press]
        res.append(value)
    names = np.asarray(names, dtype='object')
    return np.dstack((names, np.mean(np.asarray(res), axis=0)))[0]


def regularization_test(function, start, method):
    res = []
    regs = ['NoRegularization', 'L1', 'L2', 'Elastic']
    rs = [NoRegularization(), L1Regularization(), L2Regularization(), Elastic()]

    for i in range(4):
        method.set_regularization(rs[0])
        _, result = method.execute(start, function)
        res.append([regs[i], result[-1][1]])

    return res

def arithmetics_test(function, start, *methods):
    res = []
    for method in methods:
        iter, points = method.execute(start, function)
        res.append([method.name, iter * method.math_operations])
    return res

def time_test(function, start, *methods):
    res = []
    for method in methods:
        time_start = time.time_ns() / 10 ** 6
        method.execute(start, function)
        time_finish = time.time_ns() / 10 ** 6
        res.append([method.name, time_finish - time_start])
    return res

def batch_size_test(method, start, finish, data_size):
    if start >= finish or finish >= data_size : return None

    start_point = [0, 0]
    xs, ys, y_real = generate_descent_polynom(15, polynom([np.random.uniform(0, 10), np.random.uniform(0, 10)]),
                                              data_size)

    xy = np.dstack((xs, ys))[0]
    res = []

    func = MiniBatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

    for i in range(start, finish):
        func.set_batch(i)
        iterations, _ = method.execute(start_point, func)
        res.append([i, iterations])
    return res