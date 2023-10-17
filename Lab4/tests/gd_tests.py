import matplotlib.pyplot as plt
import torch

from functools import partial

from Lab4.lib_unwrapped.gradient_descent import *
from Lab4.util.graphic_util import draw_hist, plot_both
from Lab4.util.run_functions import run_cusom_and_torch, RunResult


def time_memory_test(iters, f, x, func_to_str):
    all_gds = [
        (partial(sgd, lr=0.2), partial(torch.optim.SGD, lr=0.2, momentum=0), "SGD"),
        (partial(momentum_gd, lr=0.04, momentum=0.4), partial(torch.optim.SGD, lr=0.04, momentum=0.4), "Momentum"),
        (partial(nesterov_gd, lr=0.04, momentum=0.4), partial(torch.optim.SGD, lr=0.04, momentum=0.4, nesterov=True),
         "Nesterov"),
        (partial(adagrad, lr=7), partial(torch.optim.Adagrad, lr=7), "Adagrad"),
        (partial(rmsprop, lr=0.9, beta=0.95), partial(torch.optim.RMSprop, lr=0.9, alpha=0.95), "RMSprop"),
        (partial(adam, lr=1.0, beta1=0.25, beta2=0.9), partial(torch.optim.Adam, lr=1.0, betas=(0.25, 0.9)),
         "Adam"),
    ]

    named_results = []
    for gd in all_gds:
        my_gd, torch_gd, name = gd
        custom_result = RunResult(None, 0, 0)
        torch_result = RunResult(None, 0, 0)
        for _ in range(iters):
            result_1, result_2 = run_cusom_and_torch(x, f, my_gd, torch_gd)
            custom_result.add(result_1)
            torch_result.add(result_2)
        print(name)
        named_results.append((name, custom_result))
        named_results.append((name + '(torch)', torch_result))
    draw_hist(named_results, lambda x: x[1].time_usage / iters, f'Average time for {func_to_str}')
    draw_hist(named_results, lambda x: x[1].memory_usage / iters, f'Average memory for {func_to_str}')


def main():
    def rosenbrock(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x = np.array([-5.0, 10.0])
    all_gds = [
        (partial(sgd, lr=0.2), partial(torch.optim.SGD, lr=0.2, momentum=0), "SGD"),
        (partial(momentum_gd, lr=0.04, momentum=0.4), partial(torch.optim.SGD, lr=0.04, momentum=0.4), "Momentum"),
        (partial(nesterov_gd, lr=0.04, momentum=0.4), partial(torch.optim.SGD, lr=0.04, momentum=0.4, nesterov=True),
         "Nesterov"),
        (partial(adagrad, lr=7), partial(torch.optim.Adagrad, lr=7), "Adagrad"),
        (partial(rmsprop, lr=0.9, beta=0.95), partial(torch.optim.RMSprop, lr=0.9, alpha=0.95), "RMSprop"),
        (partial(adam, lr=1.0, beta1=0.25, beta2=0.9), partial(torch.optim.Adam, lr=1.0, betas=(0.25, 0.9)),
         "Adam"),
    ]

    f = lambda x: 3 * x[0] ** 2 + 4 * x[1] ** 2 - 2
    for gd in all_gds:
        my_gd, torch_gd, name = gd
        print(name)
        plot_both(x, f, plt, my_gd, torch_gd, name)
        plt.show()

    functions = [
        (lambda x: 3 * x[0] ** 2 + 4 * x[1] ** 2 - 2, '3x^2+4y^2-2'),
        (rosenbrock, 'rosenbrock'),
        (lambda x: 0.4 * x[0] ** 2 - 2 * x[0] + 5 * x[1] ** 2, '0.4x^2-2x+5y*2'),
        (lambda x: 10 * x[0] ** 2 + 10 * x[1] ** 2, '10x^2+10y^2'),
        (lambda x: 5 * x[0] ** 2 + x[0] * x[1] + 2 * x[1] ** 2, '5x^2+xy+2y^2')
    ]
    x = np.array([-10.0, 15.0])
    iters = 50
    for func, name in functions:
        time_memory_test(iters, func, x, name)


if __name__ == "__main__":
    main()
