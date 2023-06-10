import math
from typing import Callable
import numpy as np
from numpy.linalg import norm
from abs import absNewton


class DogLeg_Met(absNewton):

    def __init__(self,
                 function: Callable,
                 max_iter: int = 1000,
                 eps: float = 10 ** (-3),
                 trust_reg: float = 1.0,
                 min_val: float = 10 ** (-9),
                 init_data: np.ndarray = None):

        super().__init__(function, max_iter, eps, min_val, init_data)
        self.trust_reg = trust_reg

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
