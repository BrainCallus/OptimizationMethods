import math
import numpy as np

def grad(f,x):
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


def line_search(x, p, f, nabl):
  alf = 1
  c1 = 1e-4
  c2 = 0.9
  fx = f(x)
  new_x = x + alf * p
  new_nabl = grad(f, new_x)
  while f(new_x) > fx+(c1 * alf * nabl.T@p) or new_nabl.T@p<=c2*nabl.T@p:
    alf*=0.5
    new_x = x+alf*p
    new_nabl = grad(f, new_x)
  return alf

def wolfe(x, eps, f):
    dim = len(x)
    steps_arg = [x]
    steps_f = [f(x)]
    H = np.eye(dim) ##approximate Hessian
    i = 0
    nabl = grad(f, x)
    while np.linalg.norm(nabl) > eps:
        i += 1
        p = -H@nabl
        alf = line_search(x,p,f,nabl)
        s = alf*p
        new_nabl = grad(f,x+alf*p)
        y = new_nabl - nabl
        y = np.array([y], dtype='float64')
        s = np.array([s], dtype='float64')
        y = np.reshape(y,(dim,1))
        s = np.reshape(s,(dim,1))
        r = 1/(y.T@s)
        l_i = (np.eye(dim)-(r*((s@(y.T)))))
        r_i = (np.eye(dim)-(r*((y@(s.T)))))
        hess_inter = l_i@H@r_i
        H = hess_inter + (r*((s@(s.T))))
        nabl = new_nabl[:]
        x = x+alf*p
        steps_arg.append(x)
        steps_f.append(f(x))
    return i, steps_arg, steps_f


def gen_learning_rate(lr_step):
    def lr_func(*args, **kwarg):
        return 0, lr_step

    return lr_func


def golden(x, eps, func, grad):
    a, b = 0, 1
    k1, k2 = (3 - math.pow(5, 0.5)) / 2, (math.pow(5, 0.5) - 1) / 2
    l1, l2 = a + k1 * (b - a), a + k2 * (b - a)
    grr = grad(x)
    xx1 = x - l1 * grr
    xx2 = x - l2 * grr
    f1, f2 = func(xx1), func(xx2)
    i = 4
    while (b - a) / 2 >= eps:
        i += 1
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
    return i, (a + b) / 2


def using_grad_vector(method, x, eps, func, grad):
    func_value = func(x)
    steps_x = [x]
    steps_y = [func_value]
    func_value_prev = func_value + eps * 5
    i = 1
    while math.fabs(func_value - func_value_prev) > eps:
        func_value_prev = func_value
        ii, lr = method(x, eps, func, grad)
        x = x - lr * grad(x)
        steps_x.append(list(x))
        steps_y.append(func_value)
        func_value = func(x)
        i += 1 + ii
    return i, steps_x, steps_y


def method_mas(method, x, iteration_numb, func, grad, eps=None):
    if eps is None:
        eps = 10
    xs = []
    ys = []
    func_calls = 0
    for i in range(1, iteration_numb):
        f_c, step_x, step_y = using_grad_vector(method, x, math.pow(eps, -i), func, grad)
        xs.append(step_x)
        ys.append(step_y)
        func_calls += f_c
    return func_calls, xs, ys
