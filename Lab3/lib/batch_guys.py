from typing import Callable


from OptimizationMethods.Lab3.lib.errors_functions import quadratic_error_func, quadratic_error_func_grad
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
                 eps: float = 10 ** (-3),
                 method: Method = None):
        super().__init__(function, max_iter, eps)
        if method is None:
            self.method = Adam(lr=time_learning_rate(100))
        else:
            self.method = method
        self.func = None

    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray = None):
        self.x = x
        self.y = y
        self.init_data = init_data
        data = np.dstack((x, y))[0]
        self.func = self.type(quadratic_error_func, quadratic_error_func_grad, data, *self.type_args())
        i, result = self.method.simple_execute(self.init_data, self.func)
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
                 method: Method = None,
                 batch_size = 50):
        super().__init__(function, max_iter, eps, method)
        self.batch = batch_size

    type = MiniBatchGD
    def type_args(self):
        return [self.batch]

class Batch(GDRegression):
    type = BatchGD
    def type_args(self):
        return []