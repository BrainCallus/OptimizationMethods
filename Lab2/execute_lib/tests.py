import time
import tracemalloc

import numpy as np

from OptimizationMethods.Lab2.lib.errors_functions import quadratic_error_func, quadratic_error_func_grad
from OptimizationMethods.Lab2.lib.functions_and_gradients import *
from OptimizationMethods.Lab2.lib.polynom_function import polynom
from OptimizationMethods.Lab2.execute_lib.regression_generation import generate_descent_polynom
from OptimizationMethods.Lab2.lib.regularization import *
from OptimizationMethods.Lab2.lib.learning_rates import *


def do_several_tests_batch_size(n, *args):
    res = []
    for i in range(n):
        print(i)  # номер теста
        time_start = time.time_ns() / 10 ** 6
        res.append(batch_size_test(*args))
        time_finish = time.time_ns() / 10 ** 6
        print(time_finish - time_start)  # затраченное время
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


def scheduling_test(function, method, real_value):
    res = []
    start = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    regs = ['Const', 'Exp', 'Time', 'Step']
    regs = np.asarray(regs, dtype='object')
    rs = [const_learning_rate(100), exp_learning_rate(100), time_learning_rate(100), step_learning_rate(100, 100)]
    for i in rs:
        method.set_lr(i)
        iter, result = method.simple_execute(start, function)
        if result[-1] > real_value + 100:
            res.append(real_value)
        else:
            res.append(result[-1])
        # res.append(iter)
    res = np.asarray(res)
    res = np.abs(res - real_value) * 10 ** 6
    return np.dstack((regs, res))[0]


def regularization_test(function, method, real_value):
    res = []
    start = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    regs = ['NoRegularization', 'L1', 'L2', 'Elastic']
    regs = np.asarray(regs, dtype='object')
    rs = [NoRegularization(), L1Regularization(), L2Regularization(), Elastic()]
    for i in rs:
        method.set_regularization(i)
        iter, result = method.execute(start, function)
        res.append(result[-1][1])
        # res.append(iter)
    res = np.asarray(res)
    res = (res - real_value)
    return np.dstack((regs, res))[0]


def regularization_test_generate(method):
    res = []
    regs = ['NoRegularization', 'L1', 'L2', 'Elastic']
    regs = np.asarray(regs, dtype='object')
    rs = [NoRegularization(), L1Regularization(), L2Regularization(), Elastic()]
    real_value = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    start = [np.random.uniform(10, 30), np.random.uniform(10, 30)]
    xs, ys, _ = generate_descent_polynom(5, polynom(real_value), 300)
    data = np.dstack((xs, ys))[0]
    function = BatchGD(quadratic_error_func, quadratic_error_func_grad, data)
    for i in rs:
        method.set_regularization(i)
        a, result = method.simple_execute(start, function)
        res.append(a)
        # print(result)
        # draw_regression(method, function, start, data, data, real_value)
    res = np.asarray(res)
    # print(res)
    # res = [np.mean(i) for i in (np.abs(res / real_value))]
    return np.dstack((regs, res))[0]


def arithmetics_test(function, start, *methods):
    res = []
    for method in methods:
        iter, points = method.execute(start, function)
        res.append([method.name, iter * method.math_operations])
    return res


def time_test(function, start, names, *methods):
    res = []
    i = 0
    for method in methods:
        time_start = time.time_ns() / 10 ** 6
        method.execute(start, function)
        time_finish = time.time_ns() / 10 ** 6
        if names is None:
            res.append([method.name, time_finish - time_start])
        else:
            res.append([names[i], time_finish - time_start])
        i += 1
    return res


def batch_size_test(method, start, finish, step, data_size):
    if start <= 0 or start >= finish or finish > data_size: return None

    start_point = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    real = [np.random.uniform(5, 10), np.random.uniform(5, 10)]
    xs, ys, y_real = \
        generate_descent_polynom(
            10,
            polynom(real),
            data_size
        )

    xy = np.dstack((xs, ys))[0]
    res = []

    func = MiniBatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

    print(real)
    print(start_point)

    for i in range(start, finish + 1, step):
        func.set_batch(i)
        time_start = time.time_ns() / 10 ** 9
        iterations, r = method.simple_execute(start_point, func)
        time_finish = time.time_ns() / 10 ** 9
        t = time_finish - time_start
        print(i, iterations, r, t)
        res.append([i, t])
    return res


def memory_test(function, start, names, *methods):
    res = []
    i = 0
    tracemalloc.start()
    for method in methods:
        tracemalloc.clear_traces()
        method.execute(start, function)
        memory_used_kb = tracemalloc.get_traced_memory()[1] / 1024
        if names is None:
            res.append([method.name, memory_used_kb])
        else:
            res.append([names[i], memory_used_kb])
        i += 1
    return res
