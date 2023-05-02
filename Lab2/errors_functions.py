import numpy as np


def quadratic_error(x, dot):
    res = 0
    for i in range(len(x)):
        res += x[i] * dot[0] ** i
    return res - dot[1]

def quadratic_error_func(x, data):
    err = np.asarray([quadratic_error(x, data[i]) for i in range(len(data))])
    return np.sum(err ** 2) / len(data)

def quadratic_error_func_grad(x, dot):
    a = quadratic_error(x, dot) * 2
    return [a * dot[0] ** i for i in range(len(x))]