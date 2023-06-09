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


""""
    тут мы, в отличие от обычного bfgs не храним матрицу, а сохраняем в очередь нужные k значений параметров
"""


def L_BFGS(x, eps, f, grad, queue_sz):
    steps_arg = [x]
    steps_f = [f(x)]
    queue_alpha = []
    queue_s_y_rho = []
    i = 1
    nabl = grad(x)
    grad_prev = nabl - nabl

    while np.linalg.norm(nabl) > eps:
        q = nabl

        for j in range(len(queue_s_y_rho)):
            s, y, rho = queue_s_y_rho[j]
            alpha = np.dot(s, q) * rho
            q -= y * alpha

        gamma = 1
        if i != 1:
            s, y, _ = queue_s_y_rho[0]
            gamma = np.dot(s, y) / (np.dot(y, y) + eps * 10 ** (-3))
        r = q * gamma

        for j in range(len(queue_s_y_rho)-1, -1, -1):
            s, y, rho = queue_s_y_rho[j]
            alpha = queue_alpha[j]
            betta = rho * np.dot(y, r)
            r += s * (alpha - betta)

        if len(queue_s_y_rho) == queue_sz:
            queue_s_y_rho.pop()
            queue_alpha.pop()

        alf = line_search(x, -r, f, nabl)
        x_prev = x
        x -= r * alf
        nabl = grad(x)
        s, y, _ = queue_s_y_rho[0]
        queue_s_y_rho.insert(0, [
            x - x_prev,
            nabl - grad_prev,
            1.0 / (np.dot(y, s) + eps * 10 ** (-3))
        ])
        queue_alpha.insert(0, alf)
        grad_prev = nabl
        steps_arg.append(x)
        steps_f.append(f(x))
        i += 1
    return i, steps_arg, steps_f
