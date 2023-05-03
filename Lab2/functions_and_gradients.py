from abc import ABC, abstractmethod
from polynom_function import *
import numpy as np

class Function:
    def __init__(self, function, gradient):
        self.function = function
        self.gradient = gradient

    def func(self, x):
        return self.function(x)

    def grad(self, x):
        return self.gradient(x)

class PolynomFunction(Function):
    def __init__(self, vector):
        function, gradient = polynom(vector)
        super().__init__(function, gradient)


class FunctionWithData(Function, ABC):
    def __init__(self, function, gradient, data):
        super().__init__(function, gradient)
        self.data = data
        self.data_size = len(data)

    def func(self, x):
        return self.function(x, self.data)

    @abstractmethod
    def grad(self, x):
        pass

class StochasticGD(FunctionWithData):
    def grad(self, x):
        dot = self.data[np.random.randint(0, self.data_size)]
        return np.asarray(self.gradient(x, dot))

class BatchGD(FunctionWithData):
    def grad(self, x):
        return np.sum([self.gradient(x, self.data[i]) for i in range(self.data_size)], axis=0) / self.data_size

class MiniBatchGD(FunctionWithData):
    def __init__(self, function, gradient, data, batch_size=250):
        super().__init__(function, gradient, data)
        self.batch = batch_size

    def grad(self, x):
        # np.random.shuffle(self.data)
        # return np.sum([grad(x, self.data[i]) for i in range(self.batch)], axis = 0) / self.batch
        return np.sum([self.gradient(x, self.data[np.random.randint(self.data_size)]) for _ in range(self.batch)],
                      axis=0) / self.batch