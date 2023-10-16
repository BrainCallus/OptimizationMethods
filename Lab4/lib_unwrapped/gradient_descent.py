from collections import deque

import numpy as np

from Lab4.util.grad_util import get_hessian, get_grad, line_search
from Lab4.lib_unwrapped.method_gd import Nesterov, Momentum, AdaGrad, RMSProp, Adam
from Lab4.lib_unwrapped.method_newton import GaussNewton, PowellDogLeg


def sgd(f, x, lr, lim=500):
    return momentum_gd(f, x, lr, 0, lim=lim)


def momentum_gd(f, x, lr, momentum, lim=500):
    return Momentum(momentum, len(x)).execute(f, x, lr, lim)


def nesterov_gd(f, x, lr, momentum, lim=500):
    return Nesterov(momentum, len(x)).execute(f, x, lr, lim)


def adagrad(f, x, lr, lim=500):
    return AdaGrad().execute(f, x, lr, lim)


def rmsprop(f, x, lr, beta, lim=500):
    return RMSProp(beta).execute(f, x, lr, lim)


def adam(f, x, lr, beta1, beta2, lim=500):
    return Adam(beta1, beta2, len(x)).execute(f, x, lr, lim)


def bfgs(f, x, lim=500):
    eps = 1e-6
    n = len(x)
    points = []
    g = None
    C = np.linalg.inv(get_hessian(f, x))
    points.append(x)
    while True:
        if g is None:
            g = get_grad(f, x)

        if np.linalg.norm(g) < eps:
            break

        p = -C @ g

        alpha = line_search(f, x, p)
        delta = p * alpha
        x = x + delta
        points.append(x)

        if len(points) > lim:
            break

        newg = get_grad(f, x)
        y = newg - g
        g = newg

        I = np.eye(n)
        rho = 1 / (y.T @ delta)
        C = (I - rho * np.outer(delta, y.T)) @ C @ (I - rho * np.outer(y, delta.T)) + \
            rho * np.outer(delta, delta.T)

    return np.array(points)


def l_bfgs(f, x, m=8, lim=500):
    eps = 1e-6
    n = len(x)
    points = []
    rho_queue = deque(maxlen=m)
    s_queue = deque(maxlen=m)
    y_queue = deque(maxlen=m)
    grad = None
    points.append(x)
    while True:
        if grad is None:
            grad = get_grad(f, x)

        if np.linalg.norm(grad) < eps:
            break

        alpha_q = []

        q = grad
        for s, rho, y in zip(reversed(s_queue), reversed(rho_queue), reversed(y_queue)):
            alpha = rho * np.outer(s.T, q)
            alpha_q.append(alpha)
            q = q - alpha @ y

        try:
            gamma = (s_queue[-1].T @ y_queue[-1]) / (y_queue[-1].T @ y_queue[-1])
            H = gamma * np.eye(n)
        except IndexError:
            H = np.linalg.inv(get_hessian(f, x))

        z = H @ q

        for s, rho, y, alpha in zip(s_queue, rho_queue, y_queue, reversed(alpha_q)):
            beta = rho * np.outer(y.T, z)
            z = z + s @ (alpha - beta)

        p = -z
        alpha = line_search(f, x, p)
        delta = p * alpha
        s_queue.append(delta)
        x = x + delta
        points.append(x)

        if len(points) > lim:
            break

        new_grad = get_grad(f, x)
        y = new_grad - grad
        y_queue.append(y)
        grad = new_grad

        rho = 1 / (y.T @ delta)
        rho_queue.append(rho)

    return np.array(points)


def gauss_newton(p, points, num_iters=100):
    return GaussNewton().execute(p, points, num_iters)


def dog_leg(p, points, trust_region=0.1, num_iters=100):
    return PowellDogLeg(trust_region).execute(p, points, num_iters)
