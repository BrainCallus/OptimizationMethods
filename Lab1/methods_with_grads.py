import math
import numpy as np

def gradient_in_the_point(f,x):
    delta = np.cbrt(np.finfo(float).eps)
    dim = len(x)
    nabl = np.zeros(dim)
    for i in range(dim):
        x_first = np.copy(x)
        x_second = np.copy(x)
        x_first[i] += delta
        x_second[i] -= delta
        nabl[i] = (f(x_first) - f(x_second))/(2*delta)
    return nabl

def line_search(x, p, f):
    nabl = gradient_in_the_point(f, x)
    alf = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    new_x = x + alf * p
    new_nabl = gradient_in_the_point(f, new_x)
    f_new_x = f(new_x)
    while f_new_x > fx + (c1 * alf * nabl.T @ p) \
            or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p) and f_new_x != fx:
        alf *= 0.5
        new_x = x + alf * p
        f_new_x = f(new_x)
        new_nabl = gradient_in_the_point(f, new_x)
    return alf

def wolfe(x, eps, f):
    dim = len(x)
    steps_arg = [x]
    steps_f = [f(x)]
    i = 1
    H = np.eye(dim)
    nabl = gradient_in_the_point(f, x)
    while np.linalg.norm(nabl) > eps:
        p = -H @ nabl
        alf = line_search(x, p, f)
        new_nabl = gradient_in_the_point(f, x + alf * p)
        y = new_nabl - nabl
        y = np.array([y], dtype='float64')
        y = np.reshape(y,(dim,1))
        s = alf*p
        s = np.array([s], dtype='float64')
        s = np.reshape(s,(dim,1))
        r = 1 / (y.T @ s)
        li = (np.eye(dim) - (r * (s @ y.T)))
        ri = (np.eye(dim) - (r * (y @ s.T)))
        hess_inter = li @ H @ ri
        H = hess_inter + (r * (s @ s.T))
        nabl = new_nabl[:]
        x = x+alf*p
        steps_arg.append(x)
        steps_f.append(f(x))
        i += 1
    return i, steps_arg, steps_f


def gen_learning_rate(lr_step):
    def lr_func(*args, **kwarg):
        return lr_step

    return lr_func


def golden(x, eps, func):
    a, b = 0, 1
    k1, k2 = (3 - math.pow(5, 0.5)) / 2, (math.pow(5, 0.5) - 1) / 2
    l1, l2 = a + k1 * (b - a), a + k2 * (b - a)
    grr = gradient_in_the_point(func, x)
    xx1 = x - l1 * grr
    xx2 = x - l2 * grr
    f1, f2 = func(xx1), func(xx2)
    while (b - a) / 2 >= eps:
        if f1 < f2:
            b, l2, f2 = l2, l1, f1
            l1 = a + k1 * (b - a)
            xx1 = x - l1 * grr
            f1 = func(xx1)
        else:
            a, l1, f1 = l1, l2, f2
            l2 = a + k2 * (b - a)
            xx2 = x - l2 * grr
            f2 = func(xx2)
    return (a + b) / 2


def using_grad_vector(method, x, eps, func):
    func_value = func(x)
    steps_x = [x]
    steps_y = [func_value]
    func_value_prev = func_value + eps * 5
    i = 1
    while math.fabs(func_value - func_value_prev) > eps:
        func_value_prev = func_value
        lr = method(x, eps, func)
        x = x - lr * gradient_in_the_point(func, x)
        steps_x.append(list(x))
        steps_y.append(func_value)
        func_value = func(x)
        i += 1
    return i, steps_x, steps_y


def method_mas(method, x, iteration_numb, func, eps=None):
    if eps is None:
        eps = 10
    xs = []
    ys = []
    iterations = []
    for i in range(1, iteration_numb):
        iter, step_x, step_y = using_grad_vector(method, x, math.pow(eps, -i), func)
        xs.append(step_x)
        ys.append(step_y)
        iterations.append(iter)
    return iterations, xs, ys
