import logging
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


class absRegression(ABC):
    logger = logging.getLogger(__name__)
    def __init__(self,
                 function: Callable,
                 max_iter: int = 1000,
                 eps: float = 10 ** (-3)):
        self.function = function
        self.max_iter = max_iter
        self.eps = eps
        self.init_data = None
        self.coefficients = None
        self.x = None
        self.y = None

    @abstractmethod
    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        """

        :param x: точки по x
        :param y: точки по y
        :param init_data: начальные коэффициенты
        :return: итерации цикла и реальных операций
        """

        pass

    def getDivergence(self) -> np.ndarray:
        return self.compute_divergence(self.coefficients)

    def compute_divergence(self, coefficients: np.ndarray) -> np.ndarray:
        y = self.function(self.x, coefficients)
        return y - self.y

    def get_computed_coefs(self) -> np.ndarray:
        return self.function(self.x, self.coefficients)