import numpy as np
import torch


def gen_points_with_source_f(count, disp, calc):
    return np.array(sorted(
        [(x := 1.0 * np.random.rand() + (i - count / 2) / (count / 3), calc(x) + np.random.normal(scale=disp))
         for i in range(count)], key=lambda x: x[0])), calc


def gen_loss(ps):
    def f(x):
        p = len(x)
        return sum((ps[:, 1] - sum(x[i] * ps[:, 0] ** i for i in range(len(x)))) ** 2) / p

    return f


def elastic_regression(p, ps, gd, alpha=0.5, lda=1):
    pt = torch.tensor(ps,requires_grad=True)
    def loss_fun(x):
        return sum((pt[:, 1] - sum(x[i] * pt[:, 0] ** i for i in range(len(x)))) ** 2) / p + lda * (
                alpha * sum(map(abs, x)) + (1 - alpha) / 2 * sum(x ** 2))

    x = np.array([0.0] * (p + 1))
    return gd(loss_fun, x)[-1]


def poly_regression(p, points, gd):
    return elastic_regression(p, points, gd, 0, 0)


def l1_regression(p, points, gd, lda=1):
    return elastic_regression(p, points, gd, 0, lda)


def l2_regression(p, points, gd, lda=1):
    return elastic_regression(p, points, gd, 1, lda)
