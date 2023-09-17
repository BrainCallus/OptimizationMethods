import math
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from OptimizationMethods.Lab3.lib.min_methods import Method

""""
в точности функция wolfe из 1 лабы, изменено только название))
"""


class absBFGS(ABC):
    def __init__(self,
                 function: Callable,
                 eps: float = 10 ** (-3)):
        self.coefficients = None
        self.init_data = None
        self.y = None
        self.x = None
        self.eps = eps
        self.func = function

    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray = None):
        self.x = x
        self.y = y
        self.init_data = init_data
        data = np.dstack((x, y))[0]
        i, result = self.execute(self.init_data, self.func)
        self.coefficients = result[0]
        return i, i

    def grad(self, x):
        delta = np.cbrt(np.finfo(float).eps)
        dim = len(x)
        nabl = np.zeros(dim)
        for i in range(dim):
            x_first = np.copy(x)
            x_second = np.copy(x)
            x_first[i] += delta
            x_second[i] -= delta
            nabl[i] = (self.func(x_first) - self.func(x_second)) / (2 * delta)
        return nabl

    def line_search(x, p, f, c1=1e-4, c2=0.9, alf=1):
        # Линейный поиск
        nabl = absBFGS.grad(f, x)
        fx = f(x)
        new_x = x + alf * p
        new_nabl = absBFGS.grad(f, new_x)
        f_new_x = f(new_x)
        while f_new_x > fx + (c1 * alf * nabl.T @ p) \
                or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p) and f_new_x != fx:
            alf *= 0.5
            new_x = x + alf * p
            f_new_x = f(new_x)
            new_nabl = absBFGS.grad(f, new_x)
        return alf

    @abstractmethod
    def execute(self, x, eps, f, grad):
        ...


class BFGS(absBFGS):
    def execute(self, x, eps, f):
        dim = len(x)
        steps_arg = [x]
        steps_f = [f(x)]
        i = 1
        aprox_hessian = np.eye(dim)
        nabl = self.grad(x)
        while np.linalg.norm(nabl) > eps:
            p = -aprox_hessian @ nabl
            alf = self.line_search(x, p, f)
            new_nabl = self.grad(x + alf * p)
            y = new_nabl - nabl
            s = alf * p
            r = 1 / (y.T @ s)
            li = (np.eye(dim) - (r * (s @ y.T)))
            ri = (np.eye(dim) - (r * (y @ s.T)))
            hess_inter = li @ aprox_hessian @ ri
            aprox_hessian = hess_inter + (r * (s @ s.T))
            nabl = new_nabl
            x += alf * p
            steps_arg.append(x)
            steps_f.append(f(x))
            i += 1
        return i, steps_arg, steps_f

class L_BFGS(absBFGS):
    def __init__(self,
                 function: Callable,
                 queue_size: int = 50):
        super().__init__(function)
        self.queue_sz = queue_size

    def execute(self, x, eps, f):
        steps_arg = [x]
        steps_f = [f(x)]
        queue_alpha = []
        queue_s_y_rho = []
        i = 1
        c1 = 10 ** (-3)
        nabl = self.grad(x)
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
                gamma = np.dot(s, y) / (np.dot(y, y) + eps * c1)
            r = q * gamma

            for j in range(len(queue_s_y_rho)-1, -1, -1):
                s, y, rho = queue_s_y_rho[j]
                alpha = queue_alpha[j]
                betta = rho * np.dot(y, r)
                r += s * (alpha - betta)

            if len(queue_s_y_rho) == self.queue_sz:
                queue_s_y_rho.pop()
                queue_alpha.pop()

            alf = self.line_search(x, -r, f, nabl)
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
