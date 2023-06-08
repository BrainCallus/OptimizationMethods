import logging
import math
from typing import Callable

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import norm

logger = logging.getLogger(__name__)


class DogLeg_Met:

    def __init__(self, function: Callable, max_iter: int = 1000, eps: float = 10 ** (-3),
                 trust_reg: float = 1.0, min_val: float = 10 ** (-9), init_data: np.ndarray = None,
                 ):
        self.function = function
        self.max_iter = max_iter  # iteration limit to prevent infinity loop
        self.eps = eps
        self.trust_reg = trust_reg
        self.min_val = min_val  # break-point, if reached, regression will be stopped
        self.coefficients = None
        self.x = None
        self.y = None
        self.init_data = None
        if init_data is not None:
            self.init_data = init_data

    def recoverCoefs(self, x: np.ndarray, y: np.ndarray, init_data: np.ndarray = None):
        """"-> np.ndarray:"""

        self.x = x
        self.y = y
        if init_data is not None:
            self.init_data = init_data

        if init_data is None:
            raise Exception("Data for computation expected")

        self.coefficients = self.init_data
        prev_square = 100
        # cur_square = 0
        epoch = 1
        iter = 0
        # while np.abs(prev_square - cur_square) >= self.eps:
        for k in range(self.max_iter):
            diverg = self.getDivergence()
            sub_iter, sigma = self.getSigma(diverg)
            iter = iter + sub_iter
            self.coefficients = self.coefficients - sigma
            cur_square = np.sqrt(np.sum(diverg ** 2))
            logger.info(f"Epoch {epoch}: CUR {cur_square}")
            delta = np.abs(prev_square - cur_square)
            if delta < self.eps:
                logger.info("Required error achieved")
                return epoch, iter
            if cur_square < self.min_val:
                logger.info("Min val for square reached.")
                return epoch, iter
            prev_square = cur_square
            epoch = epoch + 1
        logger.info("Max number of iterations reached. Regression didn't converge.")

        return epoch, iter

    # only here the difference with Gauss-Newton and additional field trust_reg
    def getSigma(self, diverg):
        sub_iters, jacobian = self.compute_jacobian(self.coefficients, step=10 ** (-3))
        sigma_gn = self.pseudo_inverse(jacobian) @ diverg
        if norm(sigma_gn) < self.trust_reg:
            return sub_iters, sigma_gn
        sigma_sd = jacobian.T @ diverg
        t = norm(sigma_sd) ** 2 / norm(jacobian @ sigma_sd) ** 2
        if t * norm(sigma_sd) > self.trust_reg:
            return sub_iters, self.trust_reg / norm(sigma_sd) * sigma_sd
        return sub_iters, t * sigma_sd + self.getS(sigma_gn, t * sigma_sd) * (sigma_gn - t * sigma_sd)

    def getS(self, sigm_gn, tsigm_sd):
        dif = sigm_gn - tsigm_sd
        scal = np.dot(tsigm_sd, dif)
        ndif = norm(dif)
        nsd = norm(tsigm_sd)
        discr = 4 * scal * scal + 4 * (self.trust_reg * self.trust_reg - nsd * nsd) * ndif ** 2
        return (-2 * scal + math.sqrt(discr)) / (2 * ndif ** 2)

    def getDivergence(self) -> np.ndarray:
        return self.compute_divergence(self.coefficients)

    def get_computed_coefs(self) -> np.ndarray:
        return self.function(self.x, self.coefficients)

    def compute_divergence(self, coefficients: np.ndarray) -> np.ndarray:
        y = self.function(self.x, coefficients)
        return y - self.y

    def compute_jacobian(self, x0: np.ndarray, step: float = 10 ** (-6)):
        y0 = self.compute_divergence(x0)
        jacobian = []
        sub_iter = 0
        for i, parameter in enumerate(x0):
            sub_iter = sub_iter + 1
            x = x0.copy()
            x[i] += step
            y = self.compute_divergence(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T
        return sub_iter, jacobian

    @staticmethod
    def pseudo_inverse(x: np.ndarray) -> np.ndarray:
        # Moore-Penrose inverse.
        return pinv(x.T @ x) @ x.T
    # (J^tJ)^-1 J^T
