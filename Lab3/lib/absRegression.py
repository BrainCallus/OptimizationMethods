import logging
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class absRegression(ABC):
    logger = logging.getLogger(__name__)

    def __init__(self,
                 max_iter: int = 1000,
                 eps: float = 10 ** (-3)):
        self.max_iter = max_iter
        self.eps = eps
        self.coefficients = None
        self.x = None
        self.y = None

    @abstractmethod
    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        ...

    def getDivergence(self) -> np.ndarray:
        return self.computeDivergence(self.coefficients)

    def computeDivergence(self,
                          coefficients: np.ndarray) -> np.ndarray:
        y = self.function(self.x, coefficients)
        return y - self.y

    def getComputedCoefficients(self) -> np.ndarray:
        return self.function(self.x, self.coefficients)
    
    def getResult(self) -> list:
        return self.coefficients

    def grad(self, func, *args):
        delta = np.cbrt(np.finfo(float).eps)
        dim = len(args[0])
        nabl = np.zeros(dim)
        for i in range(dim):
            nabl[i] = (func(args[0] + delta, *args[1:]) - func(args[0] - delta, *args[1:])) / (2 * delta)
        return nabl
