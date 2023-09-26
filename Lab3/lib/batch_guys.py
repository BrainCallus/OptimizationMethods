from typing import Callable

from OptimizationMethods.Lab3.lib.errors_functions import quadratic_error_func
from OptimizationMethods.Lab3.lib.absRegression import absRegression
from OptimizationMethods.Lab3.lib.min_methods import *
from OptimizationMethods.Lab3.lib.functions import *
from OptimizationMethods.Lab3.lib.learning_rates import *


class GDRegression(absRegression):
    @property
    @abstractmethod
    def type(self):
        ...

    @property
    @abstractmethod
    def type_args(self):
        ...

    def __init__(self,
                 function: Callable,
                 max_iter: int = 1000,
                 eps: float = 0.001,
                 method: Method = Adam(lr=time_learning_rate(100))):
        super().__init__(function, max_iter, eps)
        self.method = method
        self.func = None

    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        self.x = x
        self.y = y
        self.func = self.type(lambda a, b: quadratic_error_func(a, self.function, b),
                              lambda a, b: self.grad(quadratic_error_func, a, self.function, b),
                              np.dstack((x, y))[0], *self.type_args())
        i, result = self.method.simple_execute(init_data, self.func)
        self.coefficients = result[0]
        return i, i


class Stochastic(GDRegression):
    type = StochasticGD

    def type_args(self):
        return []


class MiniBatch(GDRegression):
    def __init__(self,
                 function: Callable,
                 max_iter: int = 1000,
                 eps: float = 10 ** (-3),
                 method: Method = Adam(lr=time_learning_rate(100)),
                 batch_size=50):
        super().__init__(function, max_iter, eps, method)
        self.batch = batch_size

    type = MiniBatchGD

    def type_args(self):
        return [self.batch]


class Batch(GDRegression):
    type = BatchGD

    def type_args(self):
        return []
