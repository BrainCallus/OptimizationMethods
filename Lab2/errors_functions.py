import numpy as np

def error(x, dot):
    return x[0] * dot[0] + x[1] - dot[1]

def error_func(x, data):
    err = np.asarray([error(x, data[i]) for i in range(len(data))])
    return np.sum(err ** 2)

def error_func_grad(x, dot):
    a = error(x, dot) * 2
    return [a * dot[0], a]