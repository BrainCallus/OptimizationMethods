from abc import abstractmethod

import numpy as np

from Lab4.util.grad_util import get_jacobian


class MethodNewton:
    def __init__(self):
        self.eps = 1e-6

    def execute(self, p, points, num_iters=100):
        coefs = np.zeros(p + 1)
        x = points[:, 0]
        y = points[:, 1]
        r = lambda w: y - np.array(sum(w[i] * x ** i for i in range(p + 1)))
        err = lambda w: sum(map(lambda ri: ri ** 2, r(w)))
        for cnt in range(num_iters):
            jacobian = get_jacobian(r, coefs)
            delta = self.get_delta(jacobian, r(coefs))
            prev_err = err(coefs)
            coefs += delta

            if abs(err(coefs) - prev_err) < 1e-5:
                break

        return coefs

    @abstractmethod
    def get_delta(self, jacobian, residuals):
        pass


class GaussNewton(MethodNewton):
    def __init__(self):
        super().__init__()

    def get_delta(self, jacobian, residuals):
        return -(np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ residuals)


class PowellDogLeg(MethodNewton):
    def __init__(self, trust_region):
        super().__init__()
        self.trust_region = trust_region

    def get_delta(self, jacobian, residuals):
        gn_delta = -np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T @ residuals
        if np.linalg.norm(gn_delta) <= self.trust_region:
            return gn_delta

        st_delta = -jacobian.T @ residuals
        if np.linalg.norm(st_delta) > self.trust_region:
            return st_delta / np.linalg.norm(st_delta) * self.trust_region

        t = (np.linalg.norm(st_delta) / np.linalg.norm(jacobian @ st_delta)) ** 2
        return t * st_delta + self.trust_region * (gn_delta - t * st_delta)
