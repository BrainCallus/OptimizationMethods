import math
import numpy as np
""""
в точности функция wolfe из 1 лабы, изменено только название))
"""

def line_search(x, p, f, grad):
    nabl = grad(x)
    alf = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    new_x = x + alf * p
    new_nabl = grad(new_x)
    f_new_x = f(new_x)
    while f_new_x > fx + (c1 * alf * nabl.T @ p) \
            or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p) and f_new_x != fx:
        alf *= 0.5
        new_x = x + alf * p
        f_new_x = f(new_x)
        new_nabl = grad(new_x)
    return alf


def BFGS(x, eps, f, grad):
    dim = len(x)
    steps_arg = [x]
    steps_f = [f(x)]
    i = 1
    aprox_hessian = np.eye(dim)
    nabl = grad(x)
    while np.linalg.norm(nabl) > eps:
        p = -aprox_hessian @ nabl
        alf = line_search(x, p, f, grad)
        new_nabl = grad(x + alf * p)
        y = new_nabl - nabl
        y = np.array([y], dtype='float64')
        y = np.reshape(y, (dim, 1))
        s = alf * p
        s = np.array([s], dtype='float64')
        s = np.reshape(s, (dim, 1))
        r = 1 / (y.T @ s)
        li = (np.eye(dim) - (r * (s @ y.T)))
        ri = (np.eye(dim) - (r * (y @ s.T)))
        hess_inter = li @ aprox_hessian @ ri
        aprox_hessian = hess_inter + (r * (s @ s.T))
        nabl = new_nabl[:]
        x = x + alf * p
        steps_arg.append(x)
        steps_f.append(f(x))
        i += 1
    return i, steps_arg, steps_f
