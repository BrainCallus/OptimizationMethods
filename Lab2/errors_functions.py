import numpy as np

def error(ab, dot):
    return ab[0] * dot[0] + ab[1] - dot[1]

def error_func(ab, xy):
    err = np.asarray([error(ab, xy[i]) for i in range(len(xy))])
    return np.sum(err ** 2)

def error_func_grad(ab, dot):
    a = error(ab, dot) * 2
    return [a * dot[0], a]