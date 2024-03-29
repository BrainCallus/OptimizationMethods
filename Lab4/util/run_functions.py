import tracemalloc
import time

import numpy as np
import torch

from Lab4.regression.regression import gen_loss

eps = 1e-6


def run_torch_gd(params, f, xt, lim=500):
    optimizer = params([xt])
    prev = xt.clone()
    points = [xt.detach().numpy().copy()]
    for epoch in range(lim):
        optimizer.zero_grad()

        output = f(xt)
        output.backward()

        optimizer.step()
        delta = xt - prev

        if np.linalg.norm(delta.detach().numpy()) < eps:
            return np.array(points)
        prev = xt.clone()

        points.append(xt.detach().numpy().copy())

    return np.array(points)


def run_and_return_result(runable, loss_func=None):
    start = time.time()
    tracemalloc.start()
    tracemalloc.clear_traces()
    result = runable()
    memory_usage_kb = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.clear_traces()
    tracemalloc.stop()
    time_usage = time.time() - start
    return RunResult(result, time_usage, memory_usage_kb, loss_func)


def run_cusom_and_torch(x, f, my_impl, torch_impl):
    xt = torch.tensor(x, requires_grad=True)
    return run_and_return_result(lambda: my_impl(f, x)), run_and_return_result(lambda: run_torch_gd(torch_impl, f, xt))


def run_scipy_method(p, points, method):
    x = np.array([0] * (p + 1))
    result = method(gen_loss(points), x)
    return result.x


def run_both_regr(p, points, my_impl, scipy_impl, loss_func=None):
    return run_and_return_result(lambda: my_impl(p, points), loss_func), run_and_return_result(
        lambda: run_scipy_method(p, points, scipy_impl), loss_func)


class RunResult:
    def __init__(self, result, time_usage, memory_usage, loss_func=None):
        self.result = result
        self.time_usage = time_usage
        self.memory_usage = memory_usage
        self.loss = 0 if loss_func is None else loss_func(result)

    def add(self, other):
        self.time_usage += other.time_usage
        self.memory_usage += other.memory_usage
        self.loss += other.loss
