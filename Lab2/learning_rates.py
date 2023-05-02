import math
from abc import ABC, abstractmethod

class learning_rate(ABC):
    def __init__(self, numb):
        self.value = numb
        self.initial_rate = numb
        self.decay = 0.05
        self.min_value = 0.001

    def get(self):
        return self.value

    def set_new(self, new_value):
        self.value = new_value

    @abstractmethod
    def change(self, *args):
        pass

class const_learning_rate(learning_rate):
    def change(self, *args):
        pass

class time_learning_rate(learning_rate):
    def change(self, *args):
        self.value = self.initial_rate / (self.decay * (args[0] + 1))
        self.value = max(self.value, self.min_value)

class step_learning_rate(learning_rate):
    def __init__(self, numb, epoch):
        super().__init__(numb)
        self.epoch = epoch

    def change(self, *args):
        self.value = self.initial_rate * \
                     math.pow(self.decay, math.floor((1 + args[0]) / self.epoch))
        self.value = max(self.value, self.min_value)

class exp_learning_rate(learning_rate):
    def change(self, *args):
        self.value = self.initial_rate * math.exp(- self.decay * args[0])
        self.value = max(self.value, self.min_value)