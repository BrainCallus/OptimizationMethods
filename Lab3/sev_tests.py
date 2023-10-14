import numpy as np
from numpy import log, cos, sin
import matplotlib.pyplot as plt
import random

from Lab3.tests import *

def rand_func(dim: int = 4):
    functions = [cos, sin, log]
    poera = [1, 2, 3, 4, 5]
    f = []
    for _ in range(dim):
        f.append([random.choice(functions), random.choice(poera)])

    def ff(x, coeffs):
        r = np.zeros(x.size)
        for t in range(len(coeffs)):
            r += (f[t][0](x) ** f[t][1]) * coeffs[t]
        return np.asarray(r)

    return ff

# mult_test_function
def mult_test_data_size(
        solvers: list,
        test_function,
        params: list,
        test_number_for_iteration: int = 100,
        noise_init: int = 10,
        noise_real: int = 100,
        noise: int = 10,
        dimensions: int = 4
):
    res = [0 for _ in params]
    real = noise_real * np.random.random(dimensions)
    ff = rand_func(dimensions)
    for solv in solvers:
        solv.function = ff

    for i in range(len(params)):
        res[i] = mult_tests_diff_data(
            solvers,
            test_function,
            ff,
            real,
            test_number_for_iteration,
            params[i],
            noise,
            noise_init,
            False
        )
    return res

# mult_test_function
def mult_test_noise(
        solvers: list,
        test_function,
        params: list,
        test_number_for_iteration: int = 100,
        noise_init: int = 10,
        noise_real: int = 100,
        data_size: int = 100,
        dimensions = 4
):
    res = [0 for _ in params]
    real = noise_real * np.random.random(dimensions)
    ff = rand_func(dimensions)
    for solv in solvers:
        solv.function = ff

    for i in range(len(params)):
        res[i] = mult_tests_diff_data(
            solvers,
            test_function,
            ff,
            real,
            test_number_for_iteration,
            data_size,
            params[i],
            noise_init,
            False
        )
    return res

# mult_test_function
def mult_test_dimensions(
        solvers: list,
        test_function,
        params: list,
        test_number_for_iteration: int = 100,
        noise_init: int = 10,
        noise_real: int = 100,
        noise: int = 10,
        data_size: int = 100
):
    res = [0 for _ in params]

    for i in range(len(params)):
        real = noise_real * np.random.random(params[i])
        ff = rand_func(params[i])
        for solv in solvers:
            solv.function = ff

        res[i] = mult_tests_diff_data(
            solvers,
            test_function,
            ff,
            real,
            test_number_for_iteration,
            data_size,
            noise,
            noise_init,
            False
        )

    return res

def mult_tests_visuals(
        solvers: list,
        test_function,
        mult_test_function,
        params: list,
        names: list,
        test_number_for_iteration: int = 100,
        noise_init: int = 10,
        noise: int = 10,
        data_size: int = 100,

):
    res = mult_test_function(
        solvers,
        test_function,
        params,
        test_number_for_iteration,
        noise_init,
        noise,
        data_size,
    )
    
    
    ax = plt.subplot()
    for i in range(len(solvers)):
        ax.plot(params, [k[i] for k in res], "-", label=names[i])
    ax.legend(prop='monospace')
    plt.show()


def mult_tests(
        solvers: list,
        test_function,
        func,
        real: list,
        prompt: str,
        test_number: int,
        same_data: bool,
        data_size: int,
        noise: int,
        noise_init: int,
        console: bool,
):  
    if console:
        print("Start of multiple tests.")
        print(prompt)
        print()
        print("Methods name:")
        for solver in solvers: 
            print(solver.__class__.__name__)
        print()
        print("Test name:")
        print(test_function.__name__)
        print()

    res = np.asarray([0 for _ in range(len(solvers))], dtype=int)

    if same_data:
        x = np.arange(1, 1 + data_size)
        y = func(x, real)
        yn = y + noise * np.random.randn(data_size)

    for i in range(test_number):
        init = noise_init * np.random.random(len(real))
        if not same_data:
            x = np.arange(1, 1 + data_size)
            y = func(x, real)
            yn = y + noise * np.random.randn(data_size)
        for i in range(len(solvers)):
            res[i] += test_function(solvers[i], init, x, yn, real)
    
    if console:
        print("Average result of", test_number, "tests:")
        print(res/test_number)
        print("Tests are over")
        print()
        print("-------------------------------------------")

    return res / test_number

def mult_tests_diff_data(
        solvers: list,
        test_function,
        func,
        real: list,
        test_number: int = 100,
        data_size: int = 100,
        noise: int = 10,
        noise_init: int = 10,
        console: bool = True
):  
    return mult_tests(
        solvers,
        test_function,
        func,
        real,
        "Data is the same, the init x is diff.",
        test_number,
        False,
        data_size,
        noise,
        noise_init,
        console
    )

def mult_tests_same_data_diff_init(
        solvers: list,
        test_function,
        func,
        real: list,
        test_number: int = 100,
        data_size: int = 100,
        noise: int = 10,
        noise_init: int = 10,
        console: bool = True
):  
    return mult_tests(
        solvers,
        test_function,
        func,
        real,
        "Data is different each time.",
        test_number,
        True,
        data_size,
        noise,
        noise_init,
        console
    )