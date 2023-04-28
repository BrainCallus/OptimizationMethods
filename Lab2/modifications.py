import math
import numpy as np
from matplotlib import pyplot as plt

def NAG(start, f, grad, lr=0.01, eps=0.03, param=[0.612]):
    gamma = param[0]
    i = 2
    x_cur = np.asarray(start)
    x_prev = x_cur - lr
    f_cur = f(x_cur)
    f_prev = f(x_prev)
    steps = [[x_prev, f_prev], [x_cur, f_cur]]
    change_prev = x_prev - lr * np.asarray(grad(x_prev))
    while math.fabs(f_cur - f_prev) > eps:
        change_curr = x_cur - lr * np.asarray(grad(x_cur))
        x_cur = change_curr + gamma * (change_curr - change_prev)
        change_prev = change_curr
        f_prev = f_cur
        f_cur = f(x_cur)
        steps.append([x_cur, f_cur])
        i = i + 1
    return i, steps

def momentum(start, f, grad, lr=0.01, eps=0.03, param=[0.612]):
    gamma = param[0]
    i = 2
    x_cur = np.asarray(start)
    x_prev = x_cur - lr
    f_cur = f(x_cur)
    f_prev = f(x_prev)
    steps = [[x_prev, f_prev], [x_cur, f_cur]]
    v = - lr*np.asarray(grad(x_cur))
    moment = gamma
    while math.fabs(f_cur - f_prev) > eps:
        change = lr * np.asarray(grad(x_cur))
        x_cur = x_cur + moment * v - change
        f_prev = f_cur
        f_cur = f(x_cur)
        v = moment * v - change
        steps.append([x_cur, f_cur])
        i = i + 1
    return i, steps

def AdaGrad(start, f, grad, lr=0.1, eps=0.005, unused=None):
    non_zero_div = 0.001
    i = 2
    x_cur = np.asarray(start)
    x_prev = x_cur - lr
    f_cur = f(x_cur)
    f_prev = f(x_prev)
    steps = [[x_prev, f_prev], [x_cur, f_cur]]
    B = np.zeros(len(x_cur))
    while math.fabs(f_cur - f_prev) > eps:
        gr = np.asarray(grad(x_cur))
        B += gr ** 2
        x_cur = x_cur - (lr / np.sqrt(B + non_zero_div)) * gr
        f_prev = f_cur
        f_cur = f(x_cur)
        steps.append([x_cur, f_cur])
        i = i + 1
    return i, steps

def RMSProp(start, f, grad, lr=0.1, eps=0.01, param=[0.9]):
    gamma = param[0]
    non_zero_div = 0.001
    i = 2
    x_cur = np.asarray(start)
    x_prev = x_cur - lr
    f_cur = f(x_cur)
    f_prev = f(x_prev)
    steps = [[x_prev, f_prev], [x_cur, f_cur]]
    B = np.zeros(len(x_cur))
    while math.fabs(f_cur - f_prev) > eps:
        gr = np.asarray(grad(x_cur))
        B = B * gamma + (1 - gamma) * (gr ** 2)
        x_cur = x_cur - (lr / np.sqrt(B + non_zero_div)) * gr
        f_prev = f_cur
        f_cur = f(x_cur)
        steps.append([x_cur, f_cur])
        i = i + 1
    return i, steps

def Adam(start, f, grad, lr=0.01, eps=0.01, param=[0.99, 0.9]):
    beta1 = param[0]
    beta2 = param[1]
    non_zero_div = 0.001
    i = 2
    x_cur = np.asarray(start)
    x_prev = x_cur - lr
    f_cur = f(x_cur)
    f_prev = f(x_prev)
    steps = [[x_prev, f_prev], [x_cur, f_cur]]
    B = np.zeros(len(x_cur))
    m = np.zeros(len(x_cur))
    v = np.zeros(len(x_cur))
    while math.fabs(f_cur - f_prev) > eps:
        gr = np.asarray(grad(x_cur))
        m = m * beta1 + (1 - beta1) * gr
        v = v * beta2 + (1 - beta2) * gr ** 2
        mm = m / (1 - beta1 ** i)
        vv = v / (1 - beta2 ** i)
        x_cur = x_cur - (lr / np.sqrt(vv) + non_zero_div) * mm
        f_prev = f_cur
        f_cur = f(x_cur)
        steps.append([x_cur, f_cur])
        i = i + 1
    return i, steps


def output_LRate(start, method, lr=None, eps=None, param=None):
    f = lambda x1: 5 * x1[0] ** 2 + 6 * x1[1] ** 2
    grad = lambda x1: [10 * x1[0], 13 * x1[1]]

    points = method(start, f, grad, lr, eps, param)
    iterations = points[0]
    points = points[1]

    xs = [i[0][0] for i in points]
    ys = [i[0][1] for i in points]

    # left, right  = min(xs), max(xs)
    # up,   bottom = max(ys), min(ys)
    # x0 = np.linspace(left - 1, right + 1, 100)
    # y0 = np.linspace(bottom - 1, up + 1, 100)
    # x, y = np.meshgrid(x0, y0)
    # plt.contour(x, y, f([x, y]), levels=sorted(set([p[1] for p in points])))

    plt.plot(xs, ys, 'o-')
    print(iterations)
    print(points[-1])
    plt.show()

def drawGraph(start, method, lr=None, eps=None, param=None):
    f = lambda x1: x1[0] ** 2
    grad = lambda x1: [2 * x1[0]]

    points = method(start, f, grad, lr, eps, param)
    iterations = points[0]
    points = points[1]

    xs = [i[0][0] for i in points]
    ys = [i[1] for i in points]

    left, right  = min(xs), max(xs)
    x0 = np.linspace(left - 1, right + 1, 100)
    y0 = [f([i]) for i in x0]


    plt.plot(x0, y0, '-')
    plt.plot(xs, ys, '.-', color='red')
    print(iterations)
    print(points[-1])
    plt.show()


Start = [-450, -10]

# output_LRate(Start, momentum, 0.01,  0.03,  [0.612])
# output_LRate(Start, NAG,      0.01,  0.03,  [0.612])
# output_LRate(Start, AdaGrad,  0.1,   0.005, [])       # лучше взять точку [-1, -1]
# output_LRate(Start, RMSProp,  0.1,   0.01,  [0.9])
# output_LRate(Start, Adam,     0.001, 0.01,  [0.9, 0.99])

Start = [-450]

# drawGraph(Start, momentum, 0.04, 0.001, [0.612])
# drawGraph(Start, NAG,      0.04, 0.001, [0.612])
# drawGraph(Start, AdaGrad,  0.1,  0.005, [])           # лучше взять точку [-1]
# drawGraph(Start, RMSProp,  0.1,  0.01,  [0.9])
# drawGraph(Start, Adam,     0.01, 0.01,  [0.99, 0.9])

''' Attention! в адаГраде идет ЛЮТОЕ накопление
квадратов, поэтому не советуется выбирать далекие
от ответа точки старта или параметры,
увеличивающие количество итераций.
Если нет ответа в течение полусекунды, вырубай, 
это надолго (если не навсегда)'''