from random import randint
from abc import ABC, abstractmethod
import numpy as np

class Function:
    def __init__(self, function, gradient, title=""):
        self.function = function
        self.gradient = gradient
        self.title = title

    def func(self, x):
        return self.function(x)

    def grad(self, x):
        return self.gradient(x)

    def set_title(self, title):
        self.title = title

    def get_title(self):
        return self.title

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
        self.batch = min(len(data), batch_size)

    def set_batch(self, batch_size):
        self.batch = batch_size


    def grad(self, x):
        # np.random.shuffle(self.data)
        # return np.sum([self.gradient(x, self.data[i]) for i in range(self.batch)], axis = 0) / self.batch
        a = randint(0, self.data_size - self.batch)
        return np.sum([self.gradient(x, self.data[i]) for i in range(a, a + self.batch)], axis = 0) / self.batch