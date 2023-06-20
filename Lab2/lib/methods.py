import math
import numpy as np
from OptimizationMethods.Lab2.lib.learning_rates import learning_rate, const_learning_rate
from OptimizationMethods.Lab2.lib.regularization import NoRegularization, Regularization
from abc import ABC, abstractmethod


class Method(ABC):
    def __init__(self, lr=None, eps=None, regularization=None, max_iterations=10000):
        if lr is None or not (isinstance(lr, learning_rate)):
            self.lr = const_learning_rate(0.01)
        else:
            self.lr = lr
        if eps is None:
            self.eps = 0.001
        else:
            self.eps = eps
        if regularization is None or not (isinstance(regularization, Regularization)):
            self.regularization = NoRegularization()
        else:
            self.regularization = regularization
        self.max_iterations = max_iterations
        self.start = None

    def set_lr(self, lr):
        if isinstance(lr, learning_rate):
            self.lr = lr
        else:
            print("lr should be learning_rate class instance")

    def set_eps(self, eps):
        self.eps = eps

    def set_max_iterations(self, n):
        self.max_iterations = n;

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
        pass

    @abstractmethod
    def change_x(self, *args):
        pass

    @property
    @abstractmethod
    def math_operations(self):
        """
        :return:
        количество математических операций
        при подсчёте изменения x
        """
        pass

    def exec(self, start, f, steps_change):
        i = 2
        x_cur = np.asarray(start)
        x_prev = x_cur - self.get_lr()
        f_cur, f_prev = self.calc_func(f, x_cur), self.calc_func(f, x_prev)
        steps = [[x_prev, f_prev], [x_cur, f_cur]]
        self.set_params(f, x_prev)
        while math.fabs(f_cur - f_prev) > self.eps and i < self.max_iterations:
            x_cur = self.change_x(f, x_cur, i)
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
        def st_change(steps, arr):
            return arr
        return self.exec(start, f, st_change)


class GD(Method):
    name = "GD"
    math_operations = 2

    def set_params(self, grad, x):
        pass

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        return x - self.get_lr() * self.calc_grad(f, x)


class NAG(Method):
    name = "Nesterov"
    math_operations = 5

    def __init__(self, gamma=0.6, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.gamma = gamma
        self.change = None

    def set_params(self, f, x):
        self.change = - self.get_lr() * self.calc_grad(f, x)


    def change_x(self, *args):
        f = args[0]
        x = args[1]
        self.change = self.gamma * self.change + \
                      self.get_lr() * self.calc_grad(f, x - self.change)
        return x - self.change
        # self.change = x - self.get_lr() * self.calc_grad(f, x)
        # return self.change + self.gamma * (self.change - temp)


class Momentum(Method):
    name = "Momentum"
    math_operations = 4

    def __init__(self, momentum=0.612, lr=None, eps=None, regularization=None):
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

    def __init__(self, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.non_zero_div = 0.0001
        self.B = None

    def set_params(self, grad, x):
        self.B = np.zeros(len(x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = self.calc_grad(f, x)
        self.B += gr ** 2
        return x - (self.get_lr() / np.sqrt(self.B + self.non_zero_div)) * gr


class RMSProp(AdaGrad):
    name = "RMSProp"
    math_operations = 9

    def __init__(self, gamma=0.9, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.gamma = gamma

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = self.calc_grad(f, x)
        self.B = self.B * self.gamma + (1 - self.gamma) * (gr ** 2)
        return x - (self.get_lr() / np.sqrt(self.B + self.non_zero_div)) * gr


class Adam(AdaGrad):
    name = "Adam"
    math_operations = 16

    def __init__(self, beta1=0.9, beta2=0.99, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None

    def set_params(self, grad, x):
        self.B = np.zeros(len(x))
        self.m = np.zeros(len(x))
        self.v = np.zeros(len(x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        i = args[2]
        gr = self.calc_grad(f, x)
        self.m = self.m * self.beta1 + (1 - self.beta1) * gr
        self.v = self.v * self.beta2 + (1 - self.beta2) * gr ** 2
        mm = self.m / (1 - self.beta1 ** i)
        vv = self.v / (1 - self.beta2 ** i)
        return x - (self.get_lr() / np.sqrt(vv + self.non_zero_div)) * mm
