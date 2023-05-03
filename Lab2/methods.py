import math
import numpy as np
from learning_rates import learning_rate, const_learning_rate
from regularization import NoRegularization, Regularization
from abc import ABC, abstractmethod

class Method(ABC):
    def __init__(self, lr=None, eps=None, regularization=None):
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

        self.start = None

    def set_lr(self, lr):
        if isinstance(lr, learning_rate):
            self.lr = lr
        else:
            print("lr should be learning_rate class instance")

    def set_eps(self, eps):
        self.eps = eps

    def set_regularization(self, regularization):
        if isinstance(regularization, Regularization):
            self.regularization = regularization
        else:
            print("regularization should be Regularization class instance")

    def get_lr(self):
        return self.lr.get()

    @staticmethod
    def calc_grad(f, x):
        return f.grad(x)

    def calc_func(self, f, x):
        return self.regularization.calc(f, x)

    @abstractmethod
    def set_params(self, grad, x):
        pass

    @abstractmethod
    def change_x(self, *args):
        pass

    def execute(self, start, f):
        i = 2
        x_cur = np.asarray(start)
        x_prev = x_cur - self.lr.get()
        f_cur, f_prev = self.calc_func(f, x_cur), self.calc_func(f, x_prev)
        steps = [[x_prev, f_prev], [x_cur, f_cur]]
        self.set_params(f, x_prev)
        while math.fabs(f_cur - f_prev) > self.eps:
            x_cur = self.change_x(f, x_cur, i)
            f_prev = f_cur
            f_cur = self.calc_func(f, x_cur)
            steps.append(np.asarray([x_cur, f_cur], dtype='object'))
            self.lr.change(i)
            i = i + 1
            # print("Func value: ", f_cur, ", iteration: ", i, ", delta: ", f_cur - f_prev)
        return i, np.asarray(steps, dtype='object')

class GD(Method):
    name = "GD"
    def set_params(self, grad, x):
        pass

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        return x - self.lr.get() * np.asarray(self.calc_grad(f, x))

class NAG(Method):
    name = "Nesterov"
    def __init__(self, gamma=0.5, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.gamma = gamma
        self.change = None

    def set_params(self, f, x):
        self.change = x - self.lr.get() * np.asarray(self.calc_grad(f, x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        temp = self.change
        self.change = x - self.lr.get() * np.asarray(self.calc_grad(f, x))
        return self.change + self.gamma * (self.change - temp)

class Momentum(Method):
    name = "Momentum"
    def __init__(self, momentum=0.812, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.v = None
        self.momentum = momentum

    def set_params(self, f, x):
        self.v = - self.lr.get() * np.asarray(self.calc_grad(f, x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        self.v = self.momentum * self.v + np.asarray(self.calc_grad(f, x))
        return x - self.lr.get() * self.v

class AdaGrad(Method):
    name = "AdaGrad"
    def __init__(self, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.non_zero_div = 0.0001
        self.B = None

    def set_params(self, grad, x):
        self.B = np.zeros(len(x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = np.asarray(self.calc_grad(f, x))
        self.B += gr ** 2
        return x - (self.lr.get() / np.sqrt(self.B + self.non_zero_div)) * gr

class RMSProp(AdaGrad):
    name = "RMSProp"
    def __init__(self, gamma = 0.9, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.gamma = gamma

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        gr = np.asarray(self.calc_grad(f, x))
        self.B = self.B * self.gamma + (1 - self.gamma) * (gr ** 2)
        return x - (self.lr.get() / np.sqrt(self.B + self.non_zero_div)) * gr

class Adam(AdaGrad):
    name = "Adam"
    def __init__(self, beta1=0.9, beta2=0.99, lr=None, eps=None, regularization=None):
        super().__init__(lr, eps, regularization)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m =None
        self.v = None

    def set_params(self, grad, x):
        self.B = np.zeros(len(x))
        self.m = np.zeros(len(x))
        self.v = np.zeros(len(x))

    def change_x(self, *args):
        f = args[0]
        x = args[1]
        i = args[2]
        gr = np.asarray(self.calc_grad(f, x))
        self.m = self.m * self.beta1 + (1 - self.beta1) * gr
        self.v = self.v * self.beta2 + (1 - self.beta2) * gr ** 2
        mm = self.m / (1 - self.beta1 ** i)
        vv = self.v / (1 - self.beta2 ** i)
        return x - (self.lr.get() / np.sqrt(vv) + self.non_zero_div) * mm

