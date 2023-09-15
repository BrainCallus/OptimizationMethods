from abc import ABC, abstractmethod
import numpy as np

class Regularization(ABC):
    @abstractmethod
    def calc(self, f, x):
        pass


class NoRegularization(Regularization):
    def calc(self, f, x):
        return f.func(x)

class L1Regularization(Regularization):
    def __init__(self, alpha = 0.075):
        self.alpha = alpha

    def calc(self, f, x):
        return f.func(x) + self.alpha * np.sum(x ** 2)

class L2Regularization(Regularization):
    def __init__(self, beta = 0.08):
        self.beta = beta

    def calc(self, f, x):
        return f.func(x) + self.beta * np.sqrt(np.sum(x ** 2))

class Elastic(Regularization):
    def __init__(self, alpha = 0.075, beta = 0.08):
        self.alpha = alpha
        self.beta = beta

    def calc(self, f, x):
        p = np.sum(x ** 2)
        return f.func(x) + self.alpha * p + self.beta * np.sqrt(p)