import matplotlib.pyplot as plt
import torch

from functools import partial

from Lab4.lib_unwrapped.gradient_descent import *
from Lab4.util.graphic_util import draw_hist
from Lab4.util.run_functions import run_cusom_and_torch, RunResult



def time_memory_test(iters, f, x, func_to_str):
    all_gds = [
        (partial(sgd, lr=0.1), partial(torch.optim.SGD, lr=0.1, momentum=0), "SGD"),
        (partial(momentum_gd, lr=0.1, momentum=0.4), partial(torch.optim.SGD, lr=0.1, momentum=0.4), "Momentum"),
        (partial(nesterov_gd, lr=0.1, momentum=0.4), partial(torch.optim.SGD, lr=0.1, momentum=0.4, nesterov=True),
         "Nesterov"),
        (partial(adagrad, lr=13), partial(torch.optim.Adagrad, lr=5), "Adagrad"),
        (partial(rmsprop, lr=2, beta=0.99), partial(torch.optim.RMSprop, lr=0.5, alpha=0.99), "RMSprop"),
        (partial(adam, lr=2.1, beta1=0.9, beta2=0.999), partial(torch.optim.Adam, lr=1, betas=(0.9, 0.999)), "Adam"),
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
    # f = lambda x: 3 * x[0] ** 2 + 0.93 * x[1] ** 2 + 6
    # arr = np.array([10.0] * 2)
    # plot_both(arr, f, plt, partial(sgd, lr=0.2), partial(torch.optim.SGD, lr=0.2, momentum=0), "SGD")
    # plt.show()

    def rosenbrock(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    # тут выводятся графики, закоментила пока, чтобы не ждать их
    # x = np.array([-0.25, -0.25])
    # all_gds = [
    #    (partial(sgd, lr=0.001), partial(torch.optim.SGD, lr=0.001, momentum=0), "SGD"),
    #    (partial(momentum_gd, lr=0.001, momentum=0.4), partial(torch.optim.SGD, lr=0.001, momentum=0.4), "Momentum"),
    #    (partial(nesterov_gd, lr=0.001, momentum=0.4), partial(torch.optim.SGD, lr=0.001, momentum=0.4, nesterov=True),
    #     "Nesterov"),
    #    (partial(adagrad, lr=0.05), partial(torch.optim.Adagrad, lr=0.05), "Adagrad"),
    #    (partial(rmsprop, lr=0.005, beta=0.99), partial(torch.optim.RMSprop, lr=0.005, alpha=0.99), "RMSprop"),
    #    (
    #    partial(adam, lr=0.01, beta1=0.9, beta2=0.999), partial(torch.optim.Adam, lr=0.01, betas=(0.9, 0.999)), "Adam"),
    # ]
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    # fig.tight_layout()
    #
    # for gd, ax in zip(all_gds, axs.flatten()):
    #    my_gd, torch_gd, name = gd
    #    print(name)
    #    plot_both(x, f, ax, my_gd, torch_gd, name)
    # plt.show()
    f = lambda x: 3 * x[0] ** 2 + 0.93 * x[1] ** 2 + 6
    x = np.array([-5.0, 15.0])
    iters = 25
    time_memory_test(iters, rosenbrock, x, 'rosenbrock function')


if __name__ == "__main__":
    main()
