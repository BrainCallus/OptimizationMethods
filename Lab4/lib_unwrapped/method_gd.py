import numpy as np
from abc import abstractmethod

from Lab4.util.grad_util import get_grad


class MethodGd:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def execute(self, func, start_x, lr, lim=500):
        ee = 1e-8
        x = start_x
        n = len(start_x)
        points = [start_x]
        while True:
            delta = self.get_delta(func, x, lr, ee, len(points))
            if np.linalg.norm(delta) < self.eps:
                break
            x = x + delta
            points.append(x)

            if len(points) >= lim:
                break
        return np.asarray(points)

    @abstractmethod
    def get_delta(self, func, x, lr, ee, length):
        pass


class Nesterov(MethodGd):
    def __init__(self, moment, dim):
        super().__init__()
        self.moment = moment
        self.v = np.array([0] * dim)

    def get_delta(self, func, x, lr, ee, length):
        self.v = self.moment * self.v + (g := get_grad(func, x))
        return - lr * (g + self.moment * self.v)


class RMSProp(MethodGd):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.s = 0

    def get_delta(self, func, x, lr, ee, length):
        gradient = get_grad(func, x)
        self.s = self.s * self.beta + (1 - self.beta) * np.dot(gradient, gradient)
        return - lr * gradient / np.sqrt(self.s + ee)


class Momentum(MethodGd):
    def __init__(self, moment, dim):
        super().__init__()
        self.moment = moment
        self.v = np.array([0] * dim)

    def get_delta(self, func, x, lr, ee, length):
        self.v = self.moment * self.v - lr * get_grad(func, x)
        return self.v


class AdaGrad(MethodGd):
    def __init__(self):
        super().__init__()
        self.G = 0

    def get_delta(self, func, x, lr, ee, length):
        grad = get_grad(func, x)
        self.G += np.dot(grad, grad)
        return - lr * grad / np.sqrt(self.G + ee)


class Adam(MethodGd):
    def __init__(self, beta1, beta2, dim):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.s = 0
        self.v = np.array([0] * dim)

    def get_delta(self, func, x, lr, ee, length):
        grad = get_grad(func, x)
        self.v = self.v * self.beta1 + (1 - self.beta1) * grad
        self.s = self.s * self.beta2 + (1 - self.beta2) * np.dot(grad, grad)
        v_ = self.v / (1 - self.beta1 ** length)
        s_ = self.s / (1 - self.beta2 ** length)
        return - lr * v_ / np.sqrt(s_ + ee)
