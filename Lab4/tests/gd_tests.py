import matplotlib.pyplot as plt
import torch

from functools import partial

from Lab4.lib_unwrapped.gradient_descent import *
from Lab4.util.run_functions import run_cusom_and_torch


def time_memory_test():
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    f = lambda x: 3 * x[0] ** 2 + 0.93 * x[1] ** 2 + 6
    x = np.array([10.0] * 2)
    iters = 25
    all_gds = [
        (partial(sgd, lr=0.1), partial(torch.optim.SGD, lr=0.1, momentum=0), "SGD"),
        (partial(momentum_gd, lr=0.1, momentum=0.4), partial(torch.optim.SGD, lr=0.1, momentum=0.4), "Momentum"),
        (partial(nesterov_gd, lr=0.1, momentum=0.4), partial(torch.optim.SGD, lr=0.1, momentum=0.4, nesterov=True),
         "Nesterov"),
        (partial(adagrad, lr=13), partial(torch.optim.Adagrad, lr=5), "Adagrad"),
        (partial(rmsprop, lr=2, beta=0.99), partial(torch.optim.RMSprop, lr=0.5, alpha=0.99), "RMSprop"),
        (partial(adam, lr=2.1, beta1=0.9, beta2=0.999), partial(torch.optim.Adam, lr=1, betas=(0.9, 0.999)), "Adam"),
    ]

    for gd, ax in zip(all_gds, axs.flatten()):
        my_gd, torch_gd, name = gd
        time_acc1 = time_acc2 = mem_acc1 = mem_acc2 = 0
        for _ in range(iters):
            result_1, result_2 = run_cusom_and_torch(x, f, my_gd, torch_gd)
            time_acc1 += result_1.time_usage
            time_acc2 += result_2.time_usage
            mem_acc1 += result_1.memory_usage
            mem_acc2 += result_2.memory_usage

        print(f"""{name}:\n"""
              f"""\tPeak memory: {mem_acc1 / iters} vs {mem_acc2 / iters}\n"""
              f"""\tTime: {time_acc1 / iters} vs {time_acc2 / iters}""")


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
    time_memory_test()


if __name__ == "__main__":
    main()
