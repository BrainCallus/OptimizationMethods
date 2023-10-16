import numpy as np
import numdifftools as ndt
import torch


def get_grad(func, x: np.ndarray):
    return ndt.Gradient(func)(x)


def numeric_grad(func, x):
    h = 1e-5
    return (func(x[:, np.newaxis] + h * np.eye(x.size)) - func(x[:, np.newaxis] - h * np.eye(x.size))) / (2 * h)


def pytorch_grad(func, x):
    x = torch.tensor(x, requires_grad=True)
    y = func(x)
    y.backward()
    return x.grad.detach().numpy()


def get_jacobian(f, x: np.ndarray):
    j = ndt.Jacobian(f)
    return j(x)


def get_hessian(f, x: np.ndarray):
    h = ndt.Hessian(f)
    return h(x)


def check_wolfe_cond(f, x: np.ndarray, alpha, direction, c1=0.1, c2=0.9):
    grad = get_grad(f, x)
    return f(x + alpha * direction) <= f(x) + alpha * c1 * np.dot(direction, grad) and \
        abs(np.dot(direction, get_grad(f, x + alpha * direction))) <= abs(c2 * np.dot(direction, grad))


def line_search(f, x: np.ndarray, direction):
    mk = 1
    start_alpha = 0.5
    for m in range(1, 10):
        alpha = start_alpha ** m
        if check_wolfe_cond(f, x, alpha, direction):
            mk = m
            break
    return start_alpha ** mk
