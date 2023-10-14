import math
import numpy as np
from OptimizationMethods.Lab3.lib.learning_rates import learning_rate, const_learning_rate
from OptimizationMethods.Lab3.lib.regs import NoRegularization, Regularization
from abc import ABC, abstractmethod


class Method(ABC):
    def __init__(self,
                 lr: learning_rate = const_learning_rate(0.01),
                 eps: float = 0.0001,
                 regularization: Regularization = NoRegularization(),
                 max_iterations: int = 1000):
        self.lr = lr
        self.eps = eps
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.start = None

    def set_lr(self,
               lr: learning_rate):
        self.lr = lr

    def set_eps(self, eps):
        self.eps = eps

    def set_max_iterations(self, n):
        self.max_iterations = n

    def set_regularization(self, regularization):
        if isinstance(regularization, Regularization):
            self.regularization = regularization
        else:
            print("regularization should be Regularization class instance")

    def get_lr(self):
        return self.lr.get()

    def get_lrName(self):
        return self.lr.__class__.__name__

    @staticmethod
    def calc_grad(f, x):
        return np.asarray(f.grad(x))

    def calc_func(self, f, x):
        return self.regularization.calc(f, x)

    @abstractmethod
    def set_params(self, grad, x):
        ...

    @abstractmethod
    def change_x(self, *args):
        ...

    @property
    @abstractmethod
    def math_operations(self):
        ...

    def exec(self, start, f, steps_change):
        i = 2
        x_cur = np.asarray(start)
        x_prev = x_cur - self.get_lr()
        f_cur, f_prev = self.calc_func(f, x_cur), self.calc_func(f, x_prev)
        steps = [[x_prev, f_prev], [x_cur, f_cur]]
        self.set_params(f, x_prev)
        while math.fabs(f_cur - f_prev) > self.eps and i < self.max_iterations:
            x_cur = self.change_x(f, x_cur)
            f_prev = f_cur
            f_cur = self.calc_func(f, x_cur)
            steps = steps_change(steps, [x_cur, f_cur])
            self.lr.change(i)
            i += 1
        self.lr.restart()
        return i, np.asarray(steps, dtype='object')

    def execute(self, start, f):
        def st_change(steps, arr):
            return steps + [arr]

        return self.exec(start, f, st_change)

    def simple_execute(self, start, f):
        def st_change(_, arr):
            return arr

        return self.exec(start, f, st_change)


class GD(Method):
    name = "GD"
    math_operations = 2

    def set_params(self, grad, x):
        ...

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        return x - self.get_lr() * self.calc_grad(f, x)


class NAG(Method):
    name = "Nesterov"
    math_operations = 5

    def __init__(self,
                 gamma: float = 0.99,
                 lr: learning_rate = const_learning_rate(100),
                 eps: float = 0.001,
                 regularization: Regularization = NoRegularization()):
        super().__init__(lr, eps, regularization)
        self.gamma = gamma
        self.change = None

    def set_params(self, f, x):
        self.change = - self.get_lr() * self.calc_grad(f, x)

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        self.change = (self.gamma * self.change +
                       self.get_lr() * self.calc_grad(f, x - self.change))
        return x - self.change
        # self.change = x - self.get_lr() * self.calc_grad(f, x)
        # return self.change + self.gamma * (self.change - temp)


class Momentum(Method):
    name = "Momentum"
    math_operations = 4

    def __init__(self,
                 momentum: float = 0.9916,
                 lr: learning_rate = const_learning_rate(100),
                 eps: float = 0.001,
                 regularization: Regularization = NoRegularization()):
        super().__init__(lr, eps, regularization)
        self.v = None
        self.momentum = momentum

    def set_params(self, f, x):
        self.v = - self.get_lr() * self.calc_grad(f, x)

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        self.v = self.momentum * self.v - self.get_lr() * self.calc_grad(f, x)
        return x + self.v


class AdaGrad(Method):
    name = "AdaGrad"
    math_operations = 7

    def __init__(self,
                 lr: learning_rate = const_learning_rate(100),
                 eps: float = 0.001,
                 regularization: Regularization = NoRegularization(),
                 max_iter: int = 1000):
        super().__init__(lr, eps, regularization, max_iter)
        self.non_zero_div = 0.0001
        self.B = None

    def set_params(self, grad, x):
        self.B = 0

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = self.calc_grad(f, x)
        self.B += gr @ gr.T
        return x - (self.get_lr() / np.sqrt(self.B + self.non_zero_div)) * gr


class RMSProp(AdaGrad):
    name = "RMSProp"
    math_operations = 9

    def __init__(self,
                 gamma: float = 0.612,
                 weight: float = 0,
                 lr: learning_rate = const_learning_rate,
                 eps: float = 0.001,
                 regularization: Regularization = NoRegularization(),
                 max_iter: int = 1000):
        super().__init__(lr, eps, regularization, max_iter)
        self.gamma = gamma
        self.weight = weight
        self.v = None

    def set_params(self, grad, x):
        self.v = 0
        self.B = np.zeros(x.size)

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = (self.calc_grad(f, x))
        self.v = self.gamma * self.v + (1 - self.gamma) * (gr ** 2)
        if self.weight != 0:
            gr += self.weight * x
        if self.gamma > 0:
            self.B = self.B * self.gamma + gr / (np.sqrt(self.v) + self.non_zero_div)
            return x - self.gamma * self.B
        else:
            return x - self.gamma * gr / (np.sqrt(self.v) + self.non_zero_div)


class Adam(AdaGrad):
    name = "Adam"
    math_operations = 16

    def __init__(self,
                 beta1: float = 0.6,
                 beta2: float = 0.6,
                 lr: learning_rate = const_learning_rate,
                 eps: float = 1e-3,
                 regularization: Regularization = NoRegularization()):
        super().__init__(lr, eps, regularization)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.i = None

    def set_params(self, grad, x):
        self.B = np.zeros(len(x))
        self.m = np.zeros(len(x))
        self.v = np.zeros(len(x))
        self.i = 0

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = self.calc_grad(f, x)
        self.i += 1
        self.m = self.m * self.beta1 + (1 - self.beta1) * gr
        self.v = self.v * self.beta2 + (1 - self.beta2) * gr ** 2
        mm = self.m / (1 - self.beta1 ** self.i)
        vv = self.v / (1 - self.beta2 ** self.i)
        return x - (self.get_lr() / np.sqrt(vv + self.non_zero_div)) * mm


class Golden(Method):
    name = "Golden"
    math_operations = 2

    def __init__(self,
                 eps: float = 0.001,
                 regularization: Regularization = NoRegularization(),
                 max_iterations: int = 1000):
        super().__init__(eps=eps, regularization=regularization, max_iterations=max_iterations)
        self.steps = None

    def set_params(self, grad, x):
        self.steps = 500

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        a, b = 0, 1
        k1, k2 = (3 - math.sqrt(5)) / 2, (math.sqrt(5) - 1) / 2
        l1, l2 = a + k1, a + k2
        grr = self.calc_grad(f, x)
        xx1 = x - l1 * grr
        xx2 = x - l2 * grr
        f1, f2 = self.calc_func(f, xx1), self.calc_func(f, xx2)
        step = 0

        while np.abs(b - a) / 2 >= self.eps and step < self.steps:
            step += 1
            if f1 < f2:
                b = l2
                l2 = l1
                f2 = f1
                l1 = a + k1 * (b - a)
                xx1 = x - l1 * grr
                f1 = self.calc_func(f, xx1)
            else:
                a = l1
                l1 = l2
                f1 = f2
                l2 = a + k2 * (b - a)
                xx2 = x - l2 * grr
                f2 = self.calc_func(f, xx2)
        return x - (a + b) / 2 * grr
