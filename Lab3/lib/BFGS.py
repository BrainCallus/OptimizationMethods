import math
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from OptimizationMethods.Lab3.lib.absRegression import absRegression
from OptimizationMethods.Lab3.lib.errors_functions import quadratic_error_func
from OptimizationMethods.Lab3.lib.functions import MiniBatchGD


class absBFGS(absRegression, ABC):
    def __init__(self,
                 function: Callable,
                 eps: float = 10 ** (-3)):
        super().__init__(function=function, eps=eps)
        self.gradient = None
        self.func = None
        self.coefficients = None
        self.y = None
        self.x = None
        self.function = function
        self.type = MiniBatchGD
        self.type_args = lambda: [80]

    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        self.x = x
        self.y = y
        self.coefficients = init_data
        self.func = self.type(lambda a, b: quadratic_error_func(a, self.function, b),
                              lambda a, b: self.grad(quadratic_error_func, a, self.function, b),
                              np.dstack((x, y))[0], *self.type_args())
        i = self.execute()
        return i, i

    def line_search(self, x, p, c1=0.001, c2=0.9, alf=1):
        nabl = self.func.grad(x)
        fx = self.func.func(x)
        new_x = x + alf * p
        new_nabl = self.func.grad(new_x)
        f_new_x = self.func.func(new_x)
        while (f_new_x > fx + (c1 * alf * nabl.T @ p)
               or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p) and f_new_x != fx):
            alf *= 0.5
            new_x = x + alf * p
            f_new_x = self.func.func(new_x)
            new_nabl = self.func.grad(new_x)
        return alf

    @abstractmethod
    def execute(self):
        ...


class BFGS(absBFGS):
    def execute(self):
        i = 1
        dim = len(self.coefficients)
        aprox_hessian = np.eye(dim)
        nabl = (self.func.grad(self.coefficients))
        delta = 100
        while delta > self.eps and i < self.max_iter:
            p = -aprox_hessian @ nabl
            alf = self.line_search(self.coefficients, p)
            s = alf * p
            new_nabl = self.func.grad(self.coefficients + s)
            y = new_nabl - nabl
            r = 1 / (y.T @ s)
            li = (np.eye(dim) - (r * (s @ y.T)))
            ri = (np.eye(dim) - (r * (y @ s.T)))
            hess_inter = li @ aprox_hessian @ ri
            aprox_hessian = hess_inter + (r * (s @ s.T))
            nabl = new_nabl
            self.coefficients += s
            delta = np.linalg.norm(s)
            i += 1
        return i

class L_BFGS(absBFGS):
    def __init__(self,
                 function: Callable,
                 queue_size: int = 50):
        super().__init__(function)
        self.queue_sz = queue_size

    def execute(self):
        queue_alpha = []
        queue_s_y_rho = []
        i = 1
        c1 = 10 ** (-3)
        nabl = self.func.grad(self.coefficients)
        grad_prev = nabl - nabl

        while np.linalg.norm(nabl) > self.eps:
            q = nabl

            for j in range(len(queue_s_y_rho)):
                s, y, rho = queue_s_y_rho[j]
                alpha = np.dot(s, q) * rho
                q -= y * alpha

            gamma = 1
            if i != 1:
                s, y, _ = queue_s_y_rho[0]
                gamma = np.dot(s, y) / (np.dot(y, y) + self.eps * c1)
            r = q * gamma

            for j in range(len(queue_s_y_rho)-1, -1, -1):
                s, y, rho = queue_s_y_rho[j]
                alpha = queue_alpha[j]
                betta = rho * np.dot(y, r)
                r += s * (alpha - betta)

            if len(queue_s_y_rho) == self.queue_sz:
                queue_s_y_rho.pop()
                queue_alpha.pop()

            alf = self.line_search(self.coefficients, -r)
            x_prev = self.coefficients
            self.coefficients -= r * alf
            nabl = self.func.grad(self.coefficients)
            s, y, _ = queue_s_y_rho[0]
            queue_s_y_rho.insert(0, [
                self.coefficients - x_prev,
                nabl - grad_prev,
                1.0 / (np.dot(y, s) + self.eps * 10 ** (-3))
            ])
            queue_alpha.insert(0, alf)
            grad_prev = nabl
            i += 1
        return i
