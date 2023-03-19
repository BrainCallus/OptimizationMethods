import math

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


def learning_rate_vector(method, x, eps, func, grad):
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
    i = 0
    for i in range(1, iteration_numb):
        f_c, step_x, step_y = learning_rate_vector(method, x, math.pow(eps, -i), func, grad)
        xs.append(step_x)
        ys.append(step_y)
        func_calls += f_c
    return func_calls, xs, ys
