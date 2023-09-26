import math
from abc import ABC, abstractmethod

class learning_rate(ABC):
    def __init__(self, numb,
                 min_value: float = 0.001,
                 decay: float = 0.05):
        self.value = numb
        self.initial_rate = numb
        self.decay = decay
        self.min_value = min_value

    def restart(self):
        self.value = self.initial_rate

    def get(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def change(self, *args):
        if self.value > self.min_value:
            self.self_change(*args)
        elif self.value < self.value:
            self.value = self.min_value

    @abstractmethod
    def self_change(self, *args):
        pass

class const_learning_rate(learning_rate):
    def self_change(self, *args):
        pass

class time_learning_rate(learning_rate):
    def self_change(self, *args):
        self.value = self.initial_rate / (self.decay * (args[0] + 1))

class step_learning_rate(learning_rate):
    def __init__(self, numb, epoch):
        super().__init__(numb)
        self.epoch = epoch

    def self_change(self, *args):
        self.value = self.initial_rate * \
                     math.pow(self.decay, math.floor((1 + args[0]) / self.epoch))

class exp_learning_rate(learning_rate):
    def self_change(self, *args):
        self.value = self.initial_rate * math.exp(- self.decay * args[0])