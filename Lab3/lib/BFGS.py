import math

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from Lab3.lib.absRegression import absRegression
from Lab3.lib.errors_functions import quadratic_error_func
from Lab3.lib.functions import *


class absBFGS(absRegression, ABC):
    def __init__(self,
                 eps: float = 10 ** (-3)):
        super().__init__(eps=eps)
        self.gradient = None
        self.func = None
        self.coefficients = None
        self.y = None
        self.x = None
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
    def __init__(self,
                 max_iter: int = 100):
        self.type = MiniBatchGD
        self.type_args = lambda: [80]
        self.eps = 1e-3
        self.max_iter = max_iter

    def execute(self):
        i = 1
        dim = len(self.coefficients)
        H = np.eye(dim)
        I = np.eye(dim)
        nabl = (self.func.grad(self.coefficients))
        c = 1e-7
        delta = 100
        while delta > self.eps and i < self.max_iter:
            p = -H @ nabl
            alf = self.line_search(self.coefficients, p)
            s = alf * p
            self.coefficients += s
            new_nabl = self.func.grad(self.coefficients)
            y = new_nabl - nabl
            p = y @ s
            ro = 1.0 / (p + c)
            A1 = I - ro * s[:, np.newaxis] * y[np.newaxis, :]
            A2 = I - ro * y[:, np.newaxis] * s[np.newaxis, :]
            H = A1 @ (H @ A2) + (ro * s[:, np.newaxis] * s[np.newaxis, :])
            delta = np.linalg.norm(nabl - new_nabl)
            nabl = new_nabl
            i += 1
        return i


class L_BFGS(absBFGS):
    def __init__(self,
                 max_iter: int = 100,
                 queue_size: int = 20):
        self.queue_sz = queue_size
        self.type = MiniBatchGD
        self.type_args = lambda: [50]
        self.eps = 1e-3
        self.max_iter = max_iter

    def execute(self):
        class node:
            def __init__(this, s, y, rho, alpha):
                this.s = s
                this.y = y
                this.rho = rho
                this.alpha = alpha
                this.next = None
                this.prev = None

        class linked_list:
            def __init__(this):
                this.max_size = self.queue_sz
                this.first = None
                this.last = None
                this.size = 0
            
            def insert(this, s, y, rho, alpha):
                n = node(s, y, rho, alpha)
                if this.size == 0:
                    this.size = 1
                    this.first = n
                    this.last = n
                elif this.size == 1:
                    this.size = 2
                    this.first = n
                    this.first.next = this.last
                    this.last.prev = this.first
                elif this.size != this.max_size:
                    this.size += 1
                    this.first.prev = n
                    n.next = this.first
                    this.first = n
                else:
                    this.first.prev = n
                    n.next = this.first
                    this.first = n
                    this.last = this.last.prev
                    this.last.next = None

        main_list = linked_list()
        i = 1
        grad = self.func.grad(self.coefficients)
        xPrev = np.zeros(len(self.coefficients))
        gradPrev = grad - grad
        ys = 100
        c = 1e-9

        while np.linalg.norm(ys) > self.eps and i < self.max_iter:
            q = self.func.grad(grad)

            nnode = main_list.last
            while nnode != None:
                al = (np.dot(nnode.s, q) * nnode.rho)
                nnode.alpha = al
                q -= al * nnode.y
                nnode = nnode.prev
            
            gamma = 1.0
            if main_list.size != 0:
                nnode = main_list.first
                ys = nnode.y
                gamma = np.dot(nnode.y, nnode.s) / (np.dot(nnode.y, nnode.y) + c)
            r = q * gamma

            nnode = main_list.first
            while nnode != None:
                r += nnode.s * (nnode.alpha - nnode.rho * np.dot(nnode.y, r))
                nnode = nnode.next

            alf = self.line_search(self.coefficients, -r)
            self.coefficients -= r * alf
            grad = self.func.grad(self.coefficients)

            if main_list.size != 0:
                main_list.insert(self.coefficients - xPrev, grad - gradPrev, 
                                 1.0 / (np.dot(main_list.first.y, main_list.first.s) + c), 
                                 alf)
            else:
                main_list.insert(self.coefficients - xPrev, grad - gradPrev, 1e-3, alf)

            xPrev = self.coefficients
            gradPrev = grad
            i += 1
            print(i, ys, main_list.first.s, self.coefficients, xPrev)
                
        return i
