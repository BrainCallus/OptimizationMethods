import math
import numpy as np
from typing import Callable
from abc import abstractmethod
from numpy.linalg import pinv, norm

from Lab3.lib.absRegression import absRegression


class absNewton(absRegression):
    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        self.x = x
        self.y = y
        self.coefficients = init_data

        prev_square = 100
        cur_square = 0
        epoch = 1
        iter = 0
        while np.abs(prev_square - cur_square) >= self.eps and epoch <= self.max_iter:
            diverg = self.getDivergence()
            sub_iter, sigma = self.getSigma(diverg)
            iter += sub_iter
            self.coefficients -= sigma
            prev_square = cur_square
            cur_square = np.sqrt(np.sum(diverg ** 2))
            epoch += 1
        return epoch, iter

    def compute_jacobian(self, x0: np.ndarray, step: float = 10 ** (-3)):
        y0 = self.computeDivergence(x0)
        jacobian = []
        sub_iter = 0
        for i, _ in enumerate(x0):
            sub_iter += 1
            x = x0.copy()
            x[i] += step
            y = self.computeDivergence(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T
        return sub_iter, jacobian

    @staticmethod
    def pseudo_inverse(x: np.ndarray) -> np.ndarray:
        return pinv(x.T @ x) @ x.T

    @abstractmethod
    def getSigma(self, diverg):
        pass


class GN_Met(absNewton):
    def getSigma(self, diverg):
        sub_iters, jacobian = self.compute_jacobian(self.coefficients)
        return sub_iters, self.pseudo_inverse(jacobian) @ diverg


class DogLeg_Met(absNewton):
    def __init__(self,
                 function: Callable,
                 max_iter: int = 1000,
                 eps: float = 10 ** (-3),
                 trust_reg: float = 1.0):
        super().__init__(function, max_iter, eps)
        self.trust_reg = trust_reg

    def getSigma(self, diverg):
        sub_iters, jacobian = self.compute_jacobian(self.coefficients)
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
